# Compiler settings
CXX = nvcc
CXXFLAGS = -Xcompiler "-Wall -pthread -fopenmp"

# Libraries
LIBS = -lsfml-graphics -lsfml-window -lsfml-system -lX11

# File names
TARGET = game_of_life
SOURCE = game.cu

all: $(TARGET)
	./$(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCE) $(LIBS)

clean:
	rm -f $(TARGET)

.PHONY: all clean