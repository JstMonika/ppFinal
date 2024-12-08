# Compiler settings
CXX = g++
CXXFLAGS = -Wall

# Libraries
LIBS = -lsfml-graphics -lsfml-window -lsfml-system -lX11 -pthread -fopenmp

# File names
TARGET = game_of_life
SOURCE = game.cpp

all: $(TARGET)
	./$(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCE) $(LIBS)

clean:
	rm -f $(TARGET)

.PHONY: all clean