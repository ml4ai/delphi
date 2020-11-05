import concurrent.futures

def f(arg):
    print("Started a task. running=%s, arg=%s" % (running, arg))
    for i in range(100000):
        pass
    print("Done")

with concurrent.futures.ThreadPoolExecutor(8) as executor:
    for arg in get_tasks():
        executor.submit(f, arg)
