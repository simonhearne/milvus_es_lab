ARG ES_TAG=9.5.5
FROM docker.elastic.co/elasticsearch/elasticsearch:${ES_TAG}
RUN bin/elasticsearch-plugin install --batch analysis-icu
