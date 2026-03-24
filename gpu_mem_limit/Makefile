CFLAGS = -shared -fPIC -O2 -Wall -Wextra -Werror -std=c17

libgpumemlimit.so: gpu_mem_limit.c
	gcc $(CFLAGS) -o $@ $< -ldl -lpthread

clean:
	rm -f libgpumemlimit.so

.PHONY: clean
