use std::fs;

use bytemuck::{Pod, Zeroable};
use log::error;
use serde::{Deserialize, Serialize};

use anyhow::Result;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Serialize, Deserialize)]
pub struct Settings {
    diffuse_rate: f32,
    evaporation_rate: f32,
    velocity: f32,
    turn_speed: f32,
    sensor_size: i32,
    sensor_angle_degrees: f32,
    sensor_distance: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            diffuse_rate: 0.5,
            evaporation_rate: 0.02,
            velocity: 1.0,
            turn_speed: 1.0,
            sensor_size: 4,
            sensor_angle_degrees: 30.0,
            sensor_distance: 15.0,
        }
    }
}

impl Settings {
    pub fn get() -> Result<Settings> {
        match fs::read_to_string("config.json") {
            Ok(str) => serde_json::from_str(&str).map_err(anyhow::Error::msg),
            Err(err) => {
                let settings = Settings::default();
                if let Err(msg) =
                    fs::write("config.json", &serde_json::to_string(&settings).unwrap())
                {
                    error!("Failed to write config.json : {:?}", msg);
                }
                Ok(settings)
            }
        }
    }
}
