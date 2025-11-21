BIN := lic
SRC := lic.cpp
INC := -I/opt/homebrew/include
LD := -L/opt/homebrew/lib -lglfw -lGLEW

$(BIN): $(SRC)
	brew install glfw glew
	clang++ $< -o $(BIN) $(INC) -std=c++17 -framework OpenGL $(LD)

clean:
	rm $(BIN)
