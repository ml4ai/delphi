import py4j.GatewayServer;

class AdditionApplication {
  def addition(first: Int, second: Int): Int = {
    return first + second;
  }
}


object Py4JServer {
  def main(args: Array[String]): Unit = {
    val app = new AdditionApplication();
    // app is now the gateway.entry_point
    val server = new GatewayServer(app);
    println("Starting the server");
    server.start();
  }
}
