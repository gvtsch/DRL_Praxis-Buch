def my_function():
    # return Beispiel
    return 1
    
def my_generator():
    # yield Beispiel
    yield 1
    yield 2
    yield 3

result = my_function()
print(result)  # Ausgabe: 1

gen = my_generator()
print(next(gen))  # Ausgabe: 1
print(next(gen))  # Ausgabe: 2
print(next(gen))  # Ausgabe: 3