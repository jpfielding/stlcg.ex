import Config

# Default Nx backend for all envs.
# Tests override to BinaryBackend explicitly in test_helper.exs if we ever
# add an alternative backend for dev/bench.
config :nx, default_backend: Nx.BinaryBackend

if File.exists?("config/#{config_env()}.exs") do
  import_config "#{config_env()}.exs"
end
