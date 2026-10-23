# Build stage: matches the local toolchain (Maven 3.9.16, JDK 26) exactly.
FROM maven:3.9.16-eclipse-temurin-26 AS build
WORKDIR /workspace

COPY pom.xml .
COPY search-common/pom.xml search-common/
COPY search-inference/pom.xml search-inference/
COPY search-retrieval/pom.xml search-retrieval/
COPY search-api/pom.xml search-api/
COPY search-ingestion/pom.xml search-ingestion/
COPY search-eval/pom.xml search-eval/
RUN mvn -q -B -pl search-api -am dependency:go-offline

COPY search-common/src search-common/src
COPY search-inference/src search-inference/src
COPY search-retrieval/src search-retrieval/src
COPY search-api/src search-api/src
RUN mvn -q -B -pl search-api -am package -DskipTests

# Runtime stage: JRE only, glibc-based (noble, not alpine) — ONNX Runtime's
# and the tokenizers JNI binding's native libraries are glibc-linked.
# uid/gid 10001, because the noble base image already uses 1000 for "ubuntu".
FROM eclipse-temurin:26-jre-noble
RUN groupadd --system --gid 10001 search && \
    useradd --system --uid 10001 --gid search --no-create-home search
WORKDIR /app
COPY --from=build /workspace/search-api/target/search-api.jar app.jar
RUN chown search:search /app/app.jar
USER 10001:10001
EXPOSE 8080
# The JRE image has no curl or wget, so probe Actuator's health endpoint with
# bash's /dev/tcp instead of adding packages. Startup loads the ONNX models,
# hence the start period.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD ["bash", "-c", "exec 3<>/dev/tcp/127.0.0.1/8080 && printf 'GET /actuator/health HTTP/1.0\\r\\n\\r\\n' >&3 && grep -q '\"status\":\"UP\"' <&3"]
ENTRYPOINT ["java", "-jar", "app.jar"]
