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

# Runtime stage: JRE only, glibc-based (jammy, not alpine) — ONNX Runtime's
# and the tokenizers JNI binding's native libraries are glibc-linked.
FROM eclipse-temurin:26-jre-jammy
RUN groupadd --system --gid 1000 search && \
    useradd --system --uid 1000 --gid search --no-create-home search
WORKDIR /app
COPY --from=build /workspace/search-api/target/search-api.jar app.jar
RUN chown search:search /app/app.jar
USER 1000:1000
EXPOSE 8080
ENTRYPOINT ["java", "-jar", "app.jar"]
