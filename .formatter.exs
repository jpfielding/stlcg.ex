# Used by "mix format"
[
  line_length: 100,
  inputs: [
    "{mix,.formatter,.credo}.exs",
    "{config,lib,test}/**/*.{ex,exs}"
  ],
  locals_without_parens: [
    # STLCG.DSL — keep formula constructions paren-free for readability
    always: 1,
    always: 2,
    eventually: 1,
    eventually: 2,
    until: 2,
    until: 3,
    then: 2,
    then: 3
  ]
]
