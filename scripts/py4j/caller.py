from py4j.java_gateway import JavaGateway
import json

gateway = JavaGateway()                   # connect to the JVM
random = gateway.jvm.java.util.Random()   # create a java.util.Random instance
number1 = random.nextInt(10)              # call the Random.nextInt method
number2 = random.nextInt(10)
print(number1, number2)

addition_app = gateway.entry_point               # get the AdditionApplication instance
# value = addition_app.addition(number1, number2)
value = addition_app.translateJson(json.dumps({"letter": "b", "number": 3.14, "bool": True}))
print(value)
