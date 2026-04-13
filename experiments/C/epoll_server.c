#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <arpa/inet.h>
#include <sys/epoll.h>

#define PORT 8080
#define MAX_EVENTS 1024
#define BUFFER_SIZE 4096

int set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

int main() {
    int server_fd, client_fd, epoll_fd;
    struct sockaddr_in addr;
    struct epoll_event event, events[MAX_EVENTS];

    server_fd = socket(AF_INET, SOCK_STREAM, 0);

    addr.sin_family = AF_INET;
    addr.sin_port = htons(PORT);
    addr.sin_addr.s_addr = INADDR_ANY;

    bind(server_fd, (struct sockaddr *)&addr, sizeof(addr));
    listen(server_fd, SOMAXCONN);

    set_nonblocking(server_fd);

    epoll_fd = epoll_create1(0);

    event.events = EPOLLIN;
    event.data.fd = server_fd;

    epoll_ctl(epoll_fd, EPOLL_CTL_ADD, server_fd, &event);

    while (1) {
        int n = epoll_wait(epoll_fd, events, MAX_EVENTS, -1);

        for (int i = 0; i < n; i++) {

            if (events[i].data.fd == server_fd) {

                client_fd = accept(server_fd, NULL, NULL);
                set_nonblocking(client_fd);

                event.events = EPOLLIN;
                event.data.fd = client_fd;

                epoll_ctl(epoll_fd, EPOLL_CTL_ADD, client_fd, &event);

            } else {

                int fd = events[i].data.fd;
                char buffer[BUFFER_SIZE];

                int bytes = read(fd, buffer, BUFFER_SIZE);

                if (bytes <= 0) {
                    close(fd);
                    continue;
                }

                const char *response =
                    "HTTP/1.1 200 OK\r\n"
                    "Content-Length: 5\r\n"
                    "Connection: close\r\n"
                    "\r\n"
                    "Hello";

                write(fd, response, strlen(response));
                close(fd);
            }
        }
    }

    close(server_fd);
}