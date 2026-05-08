FROM rockylinux/rockylinux:10 AS base

ARG REPO_NAME=stlcg.ex
ARG USERNAME=stlcg
ARG GROUPNAME=${USERNAME}
ARG USER_UID=1000
ARG USER_GID=1000

# Add ca certs so we can build the image on vpn
COPY cas.pem /etc/pki/ca-trust/source/anchors/custom_cas.pem
RUN update-ca-trust

RUN dnf install -y epel-release && dnf -y update
RUN dnf install -y \
      curl-minimal git iproute iputils jq make openssl sudo wget which zip zsh \
      gcc gcc-c++ glibc-devel autoconf automake \
      ncurses-devel openssl-devel zlib-devel \
      unzip xz tar

FROM base AS user

# Create the user
RUN groupadd --gid ${USER_GID} ${USERNAME} && \
  useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} -s /bin/bash && \
  echo ${USERNAME} ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/${USERNAME} && \
  chmod 0440 /etc/sudoers.d/${USERNAME}

########### NON ROOT FROM HERE DOWN ###########
USER ${USERNAME}

RUN set -eux && \
    mkdir -p ${HOME}/bin && \
    sudo mkdir -p /workdir/${REPO_NAME} && \
    sudo chown ${USERNAME}:${USERNAME} /workdir

# envs
ENV HOME="/home/${USERNAME}"
ENV SDK_HOME=${HOME}/sdk
ENV PATH=${HOME}/bin:${PATH}

# pretty bash
COPY --chown=${USERNAME}:${USERNAME} .bash_pretty ${HOME}/.bash_pretty
RUN set -eux && \
    echo "source ${HOME}/.bash_pretty" >> ${HOME}/.bashrc

# claude
ENV NODE_EXTRA_CA_CERTS="/etc/pki/ca-trust/source/anchors/cas.pem"
ENV PATH="${HOME}/.local/bin:${PATH}"

# claude install
RUN curl -fsSL https://claude.ai/install.sh | bash

# asdf — version manager for Erlang/Elixir
ENV ASDF_VERSION="v0.14.1"
ENV ASDF_DIR=${HOME}/.asdf
ENV ASDF_DATA_DIR=${HOME}/.asdf
RUN set -eux && \
    git clone https://github.com/asdf-vm/asdf.git ${ASDF_DIR} --branch ${ASDF_VERSION} && \
    echo ". ${ASDF_DIR}/asdf.sh" >> ${HOME}/.bashrc && \
    echo ". ${ASDF_DIR}/completions/asdf.bash" >> ${HOME}/.bashrc

# Erlang/OTP + Elixir
ENV ERLANG_VERSION="27.2"
ENV ELIXIR_VERSION="1.18.1-otp-27"
ENV KERL_BUILD_DOCS=no
ENV KERL_INSTALL_MANPAGES=no
ENV KERL_INSTALL_HTMLDOCS=no
ENV KERL_CONFIGURE_OPTIONS="--without-javac --without-wx --without-debugger --without-observer --without-et"
RUN set -eux && \
    . ${ASDF_DIR}/asdf.sh && \
    asdf plugin add erlang https://github.com/asdf-vm/asdf-erlang.git && \
    asdf plugin add elixir https://github.com/asdf-vm/asdf-elixir.git && \
    asdf install erlang ${ERLANG_VERSION} && \
    asdf install elixir ${ELIXIR_VERSION} && \
    asdf global erlang ${ERLANG_VERSION} && \
    asdf global elixir ${ELIXIR_VERSION}

ENV PATH=${ASDF_DIR}/shims:${ASDF_DIR}/bin:${PATH}

# Hex + Rebar — required for fetching deps
RUN set -eux && \
    . ${ASDF_DIR}/asdf.sh && \
    mix local.hex --force && \
    mix local.rebar --force
