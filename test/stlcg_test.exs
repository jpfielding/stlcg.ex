defmodule STLCGTest do
  use ExUnit.Case, async: true
  doctest STLCG

  test "version/0 returns the mix.exs version" do
    assert STLCG.version() =~ ~r/^\d+\.\d+\.\d+/
  end
end
