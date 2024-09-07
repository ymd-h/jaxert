with import <nixpkgs> {};
let
  hatch-config = ({
    x86_64-linux = {
      system = "x86_64-unknown-linux-gnu";
      sha256 = "172xp3q6rxqzi3fjj8h2kaqb8d1wy2p24c4glgyqka3sdnkc3dh5";
    };
    aarch64-linux = {
      system = "aarch64-unknown-linux-gnu";
      sha256 = "1qs8r0mzs0qq2m9h76639mrqyyx7flr6kxxi0m5kkzk03pm0wzfv";
    };
    x86_64-darwin = {
      system = "x86_64-apple-darwin";
      sha256 = "0hsvjq2czcxdvfdxv3xgic4x8c1x473kvdbsjs7xn4amqvd180b3";
    };
    aarch64-darwin = {
      system = "aarch64-apple-darwin";
      sha256 = "0lzxvcc85ypnffc45vzw8dk89qsnww9rx00ki5rcnzy81i9i5n8l";
    };
  })."${builtins.currentSystem}";
in {
  hatch = stdenv.mkDerivation {
    name = "hatch";
    src = (builtins.fetchurl {
      url = "https://github.com/pypa/hatch/releases/download/hatch-v1.12.0/hatch-${hatch-config.system}.tar.gz";
      sha256 = hatch-config.sha256;
    });
    phases = ["installPhase" "patchPhase"];
    installPhase = ''
      mkdir -p $out/bin
      tar -xzf $src -O > $out/bin/hatch
      chmod +x $out/bin/hatch
    '';
  };
  bash = bashInteractive;
  actionlint = actionlint;
}
