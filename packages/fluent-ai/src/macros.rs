#[macro_export]
macro_rules! define_tool {
    ($name:ident, $def:expr) => {
        pub struct $name;
        impl $crate::traits::Tool for $name {
            fn name(&self) -> &'static str {
                stringify!($name)
            }
            fn definition(&self) -> String {
                $def.into()
            }
        }
    };
}

/// HashMap-like syntax macro for fluent-ai
#[macro_export]
macro_rules! hash_map {
    { $($key:expr => $value:expr),* $(,)? } => {
        || {
            let mut map = hashbrown::HashMap::new();
            $(
                map.insert($key.to_string(), $value);
            )*
            map
        }
    };
}
