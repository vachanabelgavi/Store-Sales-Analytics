ARG FUNCTION_DIR="/app"

# Creating a base image to install all the App dependencies
FROM python:slim-buster as build-image
#FROM huggingface/transformers-pytorch-cpu as build-image
RUN apt-get update && apt-get install -y g++ make cmake unzip libcurl4-openssl-dev
ARG FUNCTION_DIR
RUN mkdir -p ${FUNCTION_DIR}
COPY main.py ${FUNCTION_DIR}
COPY helper.py ${FUNCTION_DIR}
COPY data/ ${FUNCTION_DIR}/data
COPY requirements.txt ${FUNCTION_DIR}
RUN pip uninstall --yes jupyter
RUN pip install --target ${FUNCTION_DIR} -r ${FUNCTION_DIR}/requirements.txt
RUN pip install --target ${FUNCTION_DIR} awslambdaric

# Using the above image that has the installed libraries to be used in AWS Lambda's container image 
FROM python:slim-buster
#FROM huggingface/transformers-pytorch-cpu
ARG FUNCTION_DIR
WORKDIR ${FUNCTION_DIR}
COPY --from=build-image ${FUNCTION_DIR} ${FUNCTION_DIR}
ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]
CMD [ "main.handler" ]
