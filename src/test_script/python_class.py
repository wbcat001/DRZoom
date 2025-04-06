
from abc import ABC, abstractmethod

class Human(ABC):
    @abstractmethod
    def draw(self) -> None:
        pass

class Tanaka(Human):
    def draw(self) -> None:
        print("😁")


class Yamada(Human):
    def draw(self) -> None:
        print("👶")

class Suzuki(Human):
    def write(self) -> None:
        print("こんにちは")

class AI:
    def draw(self) -> None:
        print("👽")



def run_draw(human: Human) -> None:
    human.draw()

tanaka: Tanaka = Tanaka()
yamada: Yamada = Yamada()
suzuki: Suzuki = Suzuki()
ai: AI = AI()

# run_draw(ai)
# run_draw(tanaka)

# from typing import Protocol

# class Human(Protocol):
#     def draw(self) -> None:
#         ...


# class Tanaka:
#     def draw(self) -> None:
#         print("😁")


# class Yamada:
#     def draw(self) -> None:
#         print("👶")

# class Suzuki:
#     def write(self) -> None:
#         print("こんにちは")

# class AI:
#     def draw(self) -> None:
#         print("👽")

# def run_draw(human: Human) -> None:
#     human.draw()

# tanaka: Tanaka = Tanaka()
# yamada: Yamada = Yamada()
# suzuki: Suzuki = Suzuki() 
# ai: AI = AI()

# run_draw(yamada)
# run_draw(tanaka)
# run_draw(suzuki) 
# run_draw(ai) 