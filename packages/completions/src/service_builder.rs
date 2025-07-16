{% if include_examples %}
//! Service builder example - DELETE THIS MODULE IF NOT NEEDED
//!
//! Shows how to build services using cyrup-sugars collection types.

use cyrup_sugars::{OneOrMany, ZeroOneOrMany};
use tokio::sync::oneshot;

#[derive(Debug)]
pub struct ServiceBuilder {
    name: Option<String>,
    servers: OneOrMany<String>,
    middleware: ZeroOneOrMany<String>,
}

impl ServiceBuilder {
    pub fn new() -> Self {
        ServiceBuilder {
            name: None,
            servers: OneOrMany::one("localhost:8080".to_string()),
            middleware: ZeroOneOrMany::none(),
        }
    }
    
    pub fn name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }
    
    pub fn servers(mut self, servers: OneOrMany<String>) -> Self {
        self.servers = servers;
        self
    }
    
    pub fn middleware(mut self, middleware: ZeroOneOrMany<String>) -> Self {
        self.middleware = middleware;
        self
    }
    
    pub fn build(self) -> ServiceConfig {
        ServiceConfig {
            name: self.name.unwrap_or_else(|| "unnamed-service".to_string()),
            servers: self.servers,
            middleware: self.middleware,
        }
    }
}

#[derive(Debug)]
pub struct ServiceConfig {
    pub name: String,
    pub servers: OneOrMany<String>,
    pub middleware: ZeroOneOrMany<String>,
}

pub fn demo() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Service Builder Example:");
    
    // Example: Build a service with multiple configurations
    let service_config = ServiceBuilder::new()
        .name("my-service")
        .servers(OneOrMany::many(vec!["localhost:8080".to_string(), "localhost:8081".to_string()]).unwrap())
        .middleware(ZeroOneOrMany::none().with_pushed("cors".to_string()).with_pushed("auth".to_string()))
        .build();
    
    println!("  Built service: {:?}", service_config);
    
    // Example: Handle async tasks
    let (_tx, rx) = oneshot::channel::<String>();
    let _task = cyrup_sugars::AsyncTask::new(ZeroOneOrMany::one(rx));
    
    println!("  Service ready with async task handling");
    println!();
    
    Ok(())
}
{% endif %}