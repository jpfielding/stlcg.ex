%{
  configs: [
    %{
      name: "default",
      files: %{
        included: ["lib/", "test/"],
        excluded: []
      },
      strict: true,
      parse_timeout: 5000,
      color: true,
      checks: %{
        disabled: [
          {Credo.Check.Readability.ModuleDoc, []}
        ],
        extra: []
      }
    }
  ]
}
