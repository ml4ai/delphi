import py4j.GatewayServer;
import scala.util.parsing.json._;

class AdditionApplication {
  def addition(first: Int, second: Int): Int = {
    return first + second;
  }

  def translateJson(data: String): (String, Boolean, Int) = {
    val json = JSON.parseFull(data);
    return (json("letter"), json("bool"), json("number"));
  }
}


object Py4JServer extends App {
  val app = new AdditionApplication();
  // app is now the gateway.entry_point
  val server = new GatewayServer(app);
  println("Starting the server");
  server.start();
}
