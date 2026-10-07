package com.avanti.search.api;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.ConfigurationPropertiesScan;

@SpringBootApplication
@ConfigurationPropertiesScan
public class SearchApiApplication {

    public static void main(String[] args) {
        SpringApplication.run(SearchApiApplication.class, args);
    }
}
