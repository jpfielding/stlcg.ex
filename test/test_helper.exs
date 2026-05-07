ExUnit.start(exclude: [:parity, :exla_smoke, :slow])
Application.put_env(:nx, :default_backend, Nx.BinaryBackend)
