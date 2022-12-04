all: creater parser
creater:
	gcc creater.c cJSON.c -o creater
parser:
	gcc parser.c cJSON.c -o parser

.PHONY:clean
clean:
	rm -rf creater parser *~
