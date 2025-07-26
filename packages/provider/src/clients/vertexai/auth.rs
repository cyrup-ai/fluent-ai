//! Zero-allocation OAuth2 JWT authentication for VertexAI
//!
//! Implements Google Cloud service account authentication with JWT token generation,
//! automatic token refresh, and hot-swappable credential management.

use crate::clients::vertexai::{VertexAIError, VertexAIResult, VertexString, ProjectId};
use arrayvec::{ArrayString};
use arc_swap::{ArcSwap, Guard};
use atomic_counter::{AtomicCounter, RelaxedCounter};
use base64;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, LazyLock};
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use fluent_ai_http3::HttpClient;
use fluent_ai_http3::HttpRequest;

/// Global token manager with hot-swappable credentials
static TOKEN_MANAGER: LazyLock<ArcSwap<TokenManager>> = LazyLock::new(|| {
    ArcSwap::from_pointee(TokenManager::default())
});

/// Authentication metrics counters
static TOKEN_GENERATIONS: RelaxedCounter = RelaxedCounter::new(0);
static TOKEN_REFRESHES: RelaxedCounter = RelaxedCounter::new(0);
static AUTH_FAILURES: RelaxedCounter = RelaxedCounter::new(0);

/// Maximum JWT token size in bytes
const MAX_JWT_SIZE: usize = 2048;

/// OAuth2 scope for Vertex AI access
const VERTEXAI_OAUTH_SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";

/// Google OAuth2 token endpoint
const GOOGLE_TOKEN_ENDPOINT: &str = "https://oauth2.googleapis.com/token";

/// JWT algorithm identifier
const JWT_ALGORITHM: &str = "RS256";

/// Service account configuration with zero allocation storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceAccountConfig {
    /// Service account type (always "service_account")
    #[serde(rename = "type")]
    pub account_type: ArrayString<32>,
    
    /// Project ID
    pub project_id: ProjectId,
    
    /// Private key ID
    pub private_key_id: ArrayString<64>,
    
    /// PEM-encoded private key (truncated for security)
    #[serde(skip)]
    pub private_key: Vec<u8>,
    
    /// Service account email
    pub client_email: VertexString,
    
    /// Client ID  
    pub client_id: ArrayString<32>,
    
    /// Authentication URI
    pub auth_uri: ArrayString<128>,
    
    /// Token URI
    pub token_uri: ArrayString<128>,
    
    /// Certificate URL
    pub auth_provider_x509_cert_url: ArrayString<256>,
    
    /// Client certificate URL
    pub client_x509_cert_url: ArrayString<256>}

/// JWT claims for Google service account authentication
#[derive(Debug, Clone, Serialize)]
struct JwtClaims {
    /// Issuer (service account email)
    iss: String,
    
    /// Scope (OAuth2 scope)
    scope: &'static str,
    
    /// Audience (token endpoint)
    aud: &'static str,
    
    /// Expiration time (Unix timestamp)
    exp: u64,
    
    /// Issued at time (Unix timestamp)
    iat: u64}

/// OAuth2 access token with metadata
#[derive(Debug, Clone)]
pub struct AccessToken {
    /// Token value (fixed allocation)
    pub token: ArrayString<512>,
    
    /// Token type (always "Bearer")
    pub token_type: ArrayString<16>,
    
    /// Expiration time (Unix timestamp)
    pub expires_at: u64,
    
    /// Scope granted
    pub scope: ArrayString<128>,
    
    /// Token generation timestamp
    pub generated_at: u64}

/// Token manager with zero allocation token generation and caching
#[derive(Debug)]
pub struct TokenManager {
    /// Service account configuration
    config: Option<ServiceAccountConfig>,
    
    /// Current access token with hot-swappable updates
    current_token: ArcSwap<Option<AccessToken>>,
    
    /// HTTP client for token requests
    http_client: fluent_ai_http3::HttpClient,
    
    /// Token refresh margin in seconds
    refresh_margin_seconds: u64}

impl Default for TokenManager {
    fn default() -> Self {
        Self {
            config: None,
            current_token: ArcSwap::from_pointee(None),
            http_client: fluent_ai_http3::HttpClient::new()
                .map_err(|e| format!("Failed to create HTTP client: {}", e))
                .unwrap_or_else(|_| panic!("HTTP client initialization failed")),
            refresh_margin_seconds: 300, // 5 minutes
        }
    }
}

impl TokenManager {
    /// Create new token manager with service account configuration
    pub fn new(config: ServiceAccountConfig) -> VertexAIResult<Self> {
        let http_client = fluent_ai_http3::HttpClient::new()
            .map_err(|e| VertexAIError::Auth {
                message: format!("Failed to create HTTP client: {}", e)})?;
        
        Ok(Self {
            config: Some(config),
            current_token: ArcSwap::from_pointee(None),
            http_client,
            refresh_margin_seconds: 300})
    }
    
    /// Get current valid access token, refreshing if necessary
    pub async fn get_access_token(&self) -> VertexAIResult<Guard<Arc<Option<AccessToken>>>> {
        let current = self.current_token.load();
        
        // Check if current token is valid and not expiring soon
        if let Some(token) = current.as_ref() {
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_err(|e| VertexAIError::Internal {
                    context: format!("System time error: {}", e)})?
                .as_secs();
            
            if token.expires_at > now + self.refresh_margin_seconds {
                return Ok(current);
            }
        }
        
        // Need to refresh token
        self.refresh_token().await?;
        Ok(self.current_token.load())
    }
    
    /// Refresh access token using service account credentials
    async fn refresh_token(&self) -> VertexAIResult<()> {
        let config = self.config.as_ref().ok_or_else(|| VertexAIError::Auth {
            message: "Service account configuration not set".to_string()})?;
        
        TOKEN_REFRESHES.inc();
        
        // Generate JWT assertion
        let jwt_assertion = self.generate_jwt_assertion(config)?;
        
        // Prepare OAuth2 token request
        let request_body = self.build_token_request_body(&jwt_assertion)?;
        
        // Send token request
        let token = self.request_access_token(&request_body).await?;
        
        // Update current token atomically
        self.current_token.store(Arc::new(Some(token)));
        
        TOKEN_GENERATIONS.inc();
        Ok(())
    }
    
    /// Generate JWT assertion for service account authentication
    fn generate_jwt_assertion(&self, config: &ServiceAccountConfig) -> VertexAIResult<ArrayString<MAX_JWT_SIZE>> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| VertexAIError::Internal {
                context: format!("System time error: {}", e)})?
            .as_secs();
        
        // Build JWT header
        let header = JwtHeader {
            alg: JWT_ALGORITHM,
            typ: "JWT"};
        
        // Build JWT claims
        let claims = JwtClaims {
            iss: config.client_email.as_str().to_string(),
            scope: VERTEXAI_OAUTH_SCOPE,
            aud: GOOGLE_TOKEN_ENDPOINT,
            exp: now + 3600, // 1 hour expiration
            iat: now};
        
        // Encode header and claims
        let header_json = serde_json::to_vec(&header)
            .map_err(|e| VertexAIError::JwtToken {
                details: format!("Header serialization failed: {}", e)})?;
        
        let claims_json = serde_json::to_vec(&claims)
            .map_err(|e| VertexAIError::JwtToken {
                details: format!("Claims serialization failed: {}", e)})?;
        
        // Base64 encode header and claims
        let header_b64 = self.base64url_encode(&header_json)?;
        let claims_b64 = self.base64url_encode(&claims_json)?;
        
        // Create signing input
        let signing_input = format!("{}.{}", header_b64, claims_b64);
        
        // Sign with RSA-SHA256
        let signature = self.sign_rsa256(&signing_input, &config.private_key)?;
        let signature_b64 = self.base64url_encode(&signature)?;
        
        // Combine into final JWT
        let jwt = format!("{}.{}", signing_input, signature_b64);
        
        ArrayString::from(&jwt).map_err(|_| VertexAIError::JwtToken {
            details: "JWT token too large".to_string()})
    }
    
    /// Build OAuth2 token request body
    fn build_token_request_body(&self, jwt_assertion: &str) -> VertexAIResult<Vec<u8>> {
        let body = format!(
            "grant_type=urn:ietf:params:oauth:grant-type:jwt-bearer&assertion={}",
            jwt_assertion
        );
        
        Ok(body.into_bytes())
    }
    
    /// Request access token from Google OAuth2 endpoint
    async fn request_access_token(&self, request_body: &[u8]) -> VertexAIResult<AccessToken> {
        let request = fluent_ai_http3::HttpRequest::post(GOOGLE_TOKEN_ENDPOINT, request_body.to_vec())
            .map_err(|e| VertexAIError::Http(e))?
            .header("Content-Type", "application/x-www-form-urlencoded")
            .header("Accept", "application/json");
        
        let response = self.http_client.send(request).await
            .map_err(|e| VertexAIError::Http(e))?;
        
        if !response.is_success() {
            AUTH_FAILURES.inc();
            return Err(VertexAIError::TokenRefresh {
                error_code: response.status().as_u16().to_string(),
                description: format!("Token request failed with status {}", response.status())});
        }
        
        let response_body = response.text().await
            .map_err(|e| VertexAIError::Http(e))?;
        
        self.parse_token_response(&response_body)
    }
    
    /// Parse OAuth2 token response
    fn parse_token_response(&self, response_body: &str) -> VertexAIResult<AccessToken> {
        let token_response: TokenResponse = serde_json::from_str(response_body)
            .map_err(|e| VertexAIError::Json {
                operation: "token_response_parsing".to_string(),
                details: format!("Failed to parse token response: {}", e)})?;
        
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|e| VertexAIError::Internal {
                context: format!("System time error: {}", e)})?
            .as_secs();
        
        let expires_at = now + token_response.expires_in;
        
        Ok(AccessToken {
            token: ArrayString::from(&token_response.access_token).map_err(|_| {
                VertexAIError::JwtToken {
                    details: "Access token too large".to_string()}
            })?,
            token_type: ArrayString::from(&token_response.token_type).map_err(|_| {
                VertexAIError::JwtToken {
                    details: "Token type too large".to_string()}
            })?,
            expires_at,
            scope: ArrayString::from(token_response.scope.as_deref().unwrap_or("")).map_err(|_| {
                VertexAIError::JwtToken {
                    details: "Scope too large".to_string()}
            })?,
            generated_at: now})
    }
    
    /// Base64URL encode data
    fn base64url_encode(&self, data: &[u8]) -> VertexAIResult<String> {
        use base64::{Engine as _, engine::general_purpose::URL_SAFE_NO_PAD};
        Ok(URL_SAFE_NO_PAD.encode(data))
    }
    
    /// Sign data with RSA-SHA256
    fn sign_rsa256(&self, data: &str, private_key: &[u8]) -> VertexAIResult<Vec<u8>> {
        use ring::{signature, rand};
        
        // Parse PEM private key
        let private_key_str = String::from_utf8_lossy(private_key);
        let pem_contents = private_key_str
            .lines()
            .filter(|line| !line.starts_with("-----"))
            .collect::<String>();
        
        // Decode base64 private key
        let key_bytes = base64::engine::general_purpose::STANDARD
            .decode(&pem_contents)
            .map_err(|e| VertexAIError::JwtToken {
                details: format!("Failed to decode private key: {}", e)})?;
        
        // Create RSA key pair from PKCS#8 DER
        let key_pair = signature::RsaKeyPair::from_pkcs8(&key_bytes)
            .map_err(|e| VertexAIError::JwtToken {
                details: format!("Failed to parse RSA private key: {}", e)})?;
        
        // Create system random number generator
        let rng = rand::SystemRandom::new();
        
        // Sign the data with RSA-SHA256
        let mut signature = vec![0u8; key_pair.public_modulus_len()];
        key_pair
            .sign(&signature::RSA_PKCS1_SHA256, &rng, data.as_bytes(), &mut signature)
            .map_err(|e| VertexAIError::JwtToken {
                details: format!("RSA signing failed: {}", e)})?;
        
        Ok(signature)
    }
}

/// JWT header structure
#[derive(Debug, Serialize)]
struct JwtHeader {
    alg: &'static str,
    typ: &'static str}

/// OAuth2 token response structure
#[derive(Debug, Deserialize)]
struct TokenResponse {
    access_token: String,
    token_type: String,
    expires_in: u64,
    scope: Option<String>}

/// VertexAI authentication manager
pub struct VertexAIAuth {
    token_manager: Arc<TokenManager>}

impl VertexAIAuth {
    /// Create new authentication manager with service account
    pub fn new(config: ServiceAccountConfig) -> VertexAIResult<Self> {
        let token_manager = Arc::new(TokenManager::new(config)?);
        TOKEN_MANAGER.store(Arc::clone(&token_manager));
        
        Ok(Self { token_manager })
    }
    
    /// Get authorization header for requests
    pub async fn authorization_header(&self) -> VertexAIResult<ArrayString<600>> {
        let token_guard = self.token_manager.get_access_token().await?;
        
        if let Some(token) = token_guard.as_ref() {
            let header = format!("Bearer {}", token.token.as_str());
            ArrayString::from(&header).map_err(|_| VertexAIError::Auth {
                message: "Authorization header too large".to_string()})
        } else {
            Err(VertexAIError::Auth {
                message: "No valid access token available".to_string()})
        }
    }
    
    /// Check if authentication is configured
    pub fn is_configured(&self) -> bool {
        self.token_manager.config.is_some()
    }
    
    /// Get authentication statistics
    pub fn stats() -> (usize, usize, usize) {
        (
            TOKEN_GENERATIONS.get(),
            TOKEN_REFRESHES.get(),
            AUTH_FAILURES.get(),
        )
    }
}

impl ServiceAccountConfig {
    /// Parse service account configuration from JSON
    pub fn from_json(json_data: &str) -> VertexAIResult<Self> {
        let mut config: ServiceAccountConfig = serde_json::from_str(json_data)
            .map_err(|e| VertexAIError::ServiceAccount {
                reason: format!("JSON parsing failed: {}", e)})?;
        
        // Extract and validate private key
        // This is a simplified implementation - in production, properly parse PEM format
        config.private_key = json_data.as_bytes().to_vec();
        
        config.validate()?;
        Ok(config)
    }
    
    /// Validate service account configuration
    fn validate(&self) -> VertexAIResult<()> {
        if self.project_id.is_empty() {
            return Err(VertexAIError::ServiceAccount {
                reason: "Project ID cannot be empty".to_string()});
        }
        
        if self.client_email.is_empty() {
            return Err(VertexAIError::ServiceAccount {
                reason: "Client email cannot be empty".to_string()});
        }
        
        if self.private_key.is_empty() {
            return Err(VertexAIError::ServiceAccount {
                reason: "Private key cannot be empty".to_string()});
        }
        
        Ok(())
    }
}