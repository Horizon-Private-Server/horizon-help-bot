.PHONY: milvus-up milvus-down milvus-logs

MILVUS_COMPOSE=infra/milvus/docker-compose.yml

milvus-up:
	docker compose -f $(MILVUS_COMPOSE) up -d

milvus-down:
	docker compose -f $(MILVUS_COMPOSE) down

milvus-logs:
	docker compose -f $(MILVUS_COMPOSE) logs -f --tail=200 milvus
