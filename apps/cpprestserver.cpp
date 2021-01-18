#include <iostream>
#include <served/served.hpp>

int main(int argc, const char *argv[])
{
    served::multiplexer mux;
    mux.handle("/delphi/create_model")
        .post([](served::response & res, const served::request & req) {
            //res << req.data << endl;

            std::cout << "POST req: " << req.body() << std::endl;
        });







    std::cout << "Try this example with:" << std::endl;
    std::cout << "  curl \"http://localhost:8123/delphi/create-model\"" << std::endl;

    served::net::server server("127.0.0.1", "8123", mux);
    server.run(10);

    return (EXIT_SUCCESS);
}
 