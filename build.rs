use std::{
    fs::{read_to_string, DirBuilder, OpenOptions},
    io::Write,
    path::PathBuf,
    str::FromStr,
};

const SHADER_PATHS: &[(&str, &[&str])] = &[
    (
        "src/twoplusone/softbody/collision_grid_update",
        &[
            "FILL_LOOKUP",
            "SORT_LOOKUP",
            "UPDATE_START_INDICES_1",
            "UPDATE_START_INDICES_2",
        ],
    ),
    (
        "src/twoplusone/softbody/points_norel",
        &["VERTEX_SHADER", "FRAGMENT_SHADER"],
    ),
    (
        "src/twoplusone/softbody/softbodyrk4",
        &[
            "EULER",
            "RK4STAGE_0",
            "RK4STAGE_1",
            "RK4STAGE_2",
            "RK4STAGE_3",
            "RK4STAGE_4",
        ],
    ),
    // ("src/twoplusone/worldline/raytrace", &[""]),
    (
        "src/twoplusone/worldline/worldline_updatesoftbodies",
        &["IDENTIFY_BOUNDARY"],
    ),
    // ("src/twoplusone/worldline/worldline3d", &[""]),
];

fn include_callback(
    requested_src: &str,
    _: shaderc::IncludeType,
    _: &str,
    _: usize,
) -> Result<shaderc::ResolvedInclude, String> {
    let content = std::fs::read_to_string(
        std::path::PathBuf::new()
            .join("src")
            .join("twoplusone")
            .join(requested_src),
    )
    .map_err(|e| e.to_string())?;
    Ok(shaderc::ResolvedInclude {
        content,
        resolved_name: requested_src.to_string(),
    })
}

fn spv_path(base: &PathBuf, comp: &str) -> PathBuf {
    let mut prev = base.clone();
    prev.pop();
    let _ = DirBuilder::new().create(prev.join("spv"));
    if !comp.is_empty() {
        prev.join("spv")
            .join(format!(
                "{}_{}",
                base.file_name().unwrap().to_str().unwrap(),
                comp,
            ))
            .with_extension("spv")
    } else {
        prev.join("spv")
            .join(base.file_name().unwrap().to_str().unwrap())
            .with_extension("spv")
    }
}

fn main() {
    let compiler = shaderc::Compiler::new().unwrap();
    for (path, subcomponents) in SHADER_PATHS.iter() {
        let base = PathBuf::from_str(path).unwrap();
        let glsl = base.with_extension("glsl");
        let spv = spv_path(&base, &subcomponents[0]);
        if spv
            .metadata()
            .map(|m| m.modified().unwrap() < glsl.metadata().unwrap().modified().unwrap())
            .unwrap_or(true)
        {
            let src = read_to_string(&glsl).unwrap();
            for comp in subcomponents.iter() {
                let mut opts = shaderc::CompileOptions::new().unwrap();
                opts.set_include_callback(include_callback);
                opts.add_macro_definition(comp, None);
                let spv = spv_path(&base, comp);
                let shader = compiler
                    .compile_into_spirv(
                        &src,
                        shaderc::ShaderKind::InferFromSource,
                        spv.to_str().unwrap(),
                        "main",
                        Some(&opts),
                    )
                    .unwrap();
                let mut out_file = OpenOptions::new()
                    .create(true)
                    .write(true)
                    .open(spv)
                    .unwrap();
                out_file.write_all(shader.as_binary_u8()).unwrap();
            }
        }
    }
}
