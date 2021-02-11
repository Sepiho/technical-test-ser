FROM python:3.7

MAINTAINER Sophie Sebille

ENV PATH /opt/conda/bin:$PATH

WORKDIR /app

# Create the environment:
COPY environment.yml .
RUN --quiet wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
 	/bin/bash ~/miniconda.sh -b -p /opt/conda && \
    	rm ~/miniconda.sh && \
	conda init bash && \
	conda config --set ssl_verify no && \
	conda update -n base conda  && \  
	conda env create -f environment.yml && \
	conda install -c conda-forge rdkit -n servier_dock

# The code to run when container is started:
COPY servier ./servier
COPY main.py .
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "servier_dock", "python", "main.py"]


	