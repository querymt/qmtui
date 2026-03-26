{
  description = "qmtui - TUI frontend for QueryMT agent";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-parts.url = "github:hercules-ci/flake-parts";
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = inputs @ {
    self,
    flake-parts,
    ...
  }:
    flake-parts.lib.mkFlake {inherit inputs;} {
      systems = inputs.nixpkgs.lib.systems.flakeExposed;

      perSystem = {system, ...}: let
        overlays = [inputs.rust-overlay.overlays.default];
        pkgs = import inputs.nixpkgs {
          inherit system overlays;
        };

        rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;

        cargoToml = builtins.fromTOML (builtins.readFile ./Cargo.toml);

        qmtui = pkgs.rustPlatform.buildRustPackage {
          pname = "qmtui";
          version = cargoToml.package.version;
          src = ./.;
          cargoLock = {
            lockFile = ./Cargo.lock;
          };
          nativeBuildInputs = [
            pkgs.pkg-config
          ];
          buildInputs = [
            pkgs.openssl
          ];
          auditable = false;
          doCheck = false;
          installPhase = ''
            runHook preInstall
            mkdir -p $out/bin
            install -Dm755 target/${pkgs.stdenv.hostPlatform.rust.rustcTarget}/release/qmtui $out/bin/qmtui
            runHook postInstall
          '';
        };
      in {
        packages = {
          qmtui = qmtui;
          default = qmtui;
        };

        apps = {
          qmtui = {
            type = "app";
            program = "${self.packages.${system}.qmtui}/bin/qmtui";
          };
          default = {
            type = "app";
            program = "${self.packages.${system}.qmtui}/bin/qmtui";
          };
        };

        devShells.default = pkgs.mkShell {
          packages = [
            rustToolchain
            pkgs.pkg-config
            pkgs.openssl
          ];

          shellHook = ''
            export PS1="(dev:qmtui) $PS1"
          '';
        };
      };
    };
}
