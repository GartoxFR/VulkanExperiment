use std::sync::Arc;

use log::{info};
use vulkano::device::{Device, DeviceCreateInfo, Queue, QueueCreateInfo};
use vulkano::VulkanLibrary;

use anyhow::{Ok, Result};
use vulkano::instance::{Instance, InstanceCreateInfo};

pub fn init_vulkan() -> Result<(Arc<Device>, Arc<Queue>)> {
    let library = VulkanLibrary::new().expect("No local Vulkan library/DLL");
    let required_extensions = vulkano_win::required_extensions(&library);
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            enabled_extensions: required_extensions,
            ..Default::default()
        },
    )
    .expect("Failed to create instance");

    let physical = instance
        .enumerate_physical_devices()?
        .next()
        .expect("No available device");

    info!("{:?}", physical.properties().device_name);

    let queue_family_index = physical
        .queue_family_properties()
        .iter()
        .enumerate()
        .position(|(_, q)| q.queue_flags.graphics)
        .expect("Couldn't find a graphical queue family") as u32;

    let (device, mut queues) = Device::new(
        physical,
        DeviceCreateInfo {
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )?;

    let queue = queues.next().unwrap();

    Ok((device, queue))
}
