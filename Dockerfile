FROM ubuntu:18.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    curl build-essential bison flex libfl-dev libgmp-dev \
    python3.8 python3-pip python3.8-dev \
    && rm -rf /var/lib/apt/lists/*

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
ENV PYTHONIOENCODING=utf-8

# Create an app user so our program doesn't run as root.
RUN groupadd -r app &&\
    useradd -r -g app -d /home/app -s /sbin/nologin -c "Docker image user" app

# Set the home directory to our app user's home.
ENV HOME=/home/app
ENV APP_HOME=/home/app/sympais

## SETTING UP THE APP ##
RUN mkdir -p $APP_HOME
WORKDIR $APP_HOME

# Copy in the application code.
ADD . $APP_HOME
# install realpaver
RUN sh third_party/Realpaver/build.sh

# Chown all the files to the app user.
RUN chown -R app:app $HOME

# Change to the app user.
USER app
ENV PATH=/home/app/.local/bin:${PATH}

# Install Python dependencies with Poetry
RUN python3.8 -m pip install --user --no-cache -U setuptools wheel pip

RUN curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python3.8 -

ENV PATH=${HOME}/.poetry/bin:${PATH}
RUN poetry config virtualenvs.path ~/.virtualenvs
RUN poetry env use python3.8
RUN poetry install --no-interaction --no-ansi
RUN poetry env info
CMD poetry run python
