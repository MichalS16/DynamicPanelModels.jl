using Documenter
using DynamicPanelModels

DocMeta.setdocmeta!(DynamicPanelModels, :DocTestSetup, :(using DynamicPanelModels); recursive=true)

cp(
    joinpath(@__DIR__, "..", "CHANGELOG.md"),
    joinpath(@__DIR__, "src", "changelog.md");
    force=true,
)

makedocs(;
    modules=[DynamicPanelModels],
    authors="Michal Smieško",
    sitename="DynamicPanelModels.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://MichalS16.github.io/DynamicPanelModels.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "guide.md",
        "API Reference" => "api.md",
        "Changelog" => "changelog.md",
    ],
    warnonly=[:missing_docs],
)

deploydocs(;
    repo="github.com/MichalS16/DynamicPanelModels.jl",
    devbranch="main",
)
