.PHONY: build install run test help clean

requirements:
	pip freeze > requirements.txt
install:
	pip install -r requirements.txt
run:
	python main.py
test:
	python tester.py
execute:
	./dist/$(NAME).exe
help:
	@echo "Comandos disponíveis:"
	@echo "  make build   - Prepara o ambiente (instala dependências)"
	@echo "  make install - Instala as dependências"
	@echo "  make run     - Executa o programa principal"
	@echo "  make test    - Executa os testes"
	@echo "  make help    - Mostra esta mensagem"
	@echo "  make clean   - Remove arquivos gerados"

clean:
	Remove-Item -Recurse -Force build,dist,*.spec -ErrorAction SilentlyContinue
