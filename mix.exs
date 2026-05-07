defmodule STLCG.MixProject do
  use Mix.Project

  @version "0.1.0"
  @source_url "https://github.com/jpfielding/stlcg.ex"

  def project do
    [
      app: :stlcg,
      version: @version,
      elixir: "~> 1.18",
      elixirc_paths: elixirc_paths(Mix.env()),
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description: description(),
      package: package(),
      docs: docs(),
      name: "STLCG",
      source_url: @source_url,
      test_coverage: [tool: ExCoveralls],
      preferred_cli_env: [
        coveralls: :test,
        "coveralls.html": :test,
        "coveralls.json": :test,
        docs: :docs
      ],
      dialyzer: [
        plt_add_apps: [:ex_unit, :mix],
        flags: [:error_handling, :unknown, :underspecs]
      ]
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  defp deps do
    [
      {:nx, "~> 0.11"},
      {:jason, "~> 1.4"},
      {:ex_doc, "~> 0.34", only: [:dev, :docs], runtime: false},
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false},
      {:stream_data, "~> 1.1", only: [:dev, :test]},
      {:excoveralls, "~> 0.18", only: :test}
    ]
  end

  defp description do
    "An idiomatic Elixir port of Stanford ASL's STLCG — " <>
      "differentiable robustness of Signal Temporal Logic formulas via Nx."
  end

  defp package do
    [
      licenses: ["MIT"],
      links: %{
        "GitHub" => @source_url,
        "Upstream (StanfordASL/stlcg)" => "https://github.com/StanfordASL/stlcg"
      },
      files: ~w(lib mix.exs README.md LICENSE NOTICE CHANGELOG.md)
    ]
  end

  defp docs do
    [
      main: "readme",
      extras: [
        "README.md",
        "docs/semantics.md",
        "CHANGELOG.md",
        "NOTICE",
        "LICENSE"
      ],
      groups_for_extras: [
        Guides: ["README.md", "docs/semantics.md"],
        Project: ["CHANGELOG.md", "NOTICE", "LICENSE"]
      ],
      source_ref: "v#{@version}",
      formatters: ["html"],
      groups_for_modules: [
        "Public API": [STLCG, STLCG.DSL, STLCG.Expression, STLCG.Formula],
        Predicates: [
          STLCG.LessThan,
          STLCG.GreaterThan,
          STLCG.Equal,
          STLCG.Identity,
          STLCG.Predicates
        ],
        Logical: [STLCG.And, STLCG.Or, STLCG.Not, STLCG.Implies, STLCG.Logical],
        Temporal: [
          STLCG.Always,
          STLCG.Eventually,
          STLCG.Until,
          STLCG.Then,
          STLCG.Integral1d,
          STLCG.Temporal,
          STLCG.UntilThen,
          STLCG.Integral
        ],
        Aggregation: [STLCG.Aggregation],
        "Testing & fixtures": [STLCG.Fixtures]
      ]
    ]
  end
end
