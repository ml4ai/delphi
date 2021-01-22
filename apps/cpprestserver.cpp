#include <iostream>
#include <served/served.hpp>
#include <nlohmann/json.hpp>
//#include "dbg"


int main(int argc, const char *argv[])
{

    served::multiplexer mux;
    mux.handle("/delphi/create-model")
        .post([](served::response & res, const served::request & req) {
            //res << req.data << endl;

            auto json_data = nlohmann::json::parse(req.body());
            //std::cout << "POST req: " << json_data << std::endl;
            res << json_data.dump();
        });







    std::cout << "Try this example with:" << std::endl;
    std::cout << "curl -X POST \"http://localhost:8123/delphi/create-model\" -d @test.json --header \"Content-Type: application/json\" " << std::endl;

    served::net::server server("127.0.0.1", "8123", mux);
    server.run(10);

    return (EXIT_SUCCESS);
}
 