using Documenter
using DynamicPanelModels

DocMeta.setdocmeta!(DynamicPanelModels, :DocTestSetup, :(using DynamicPanelModels); recursive=true)

makedocs(;
    modules=[DynamicPanelModels],
    authors="Michal Smieško",
    sitename="DynamicPanelModels.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://MichalS16.github.io/DynamicPanelModels",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo="github.com/MichalS16/DynamicPanelModels",
    devbranch="main",
)
