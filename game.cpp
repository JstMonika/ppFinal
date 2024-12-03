#include <SFML/Graphics.hpp>
#include <vector>
#include <chrono>
#include <thread>

class GameOfLife {
private:
    int width, height;
    std::vector<std::vector<bool>> grid;
    std::vector<std::vector<bool>> nextGrid;
    sf::RenderWindow window;
    float cellSize = 10.0f;
    float updateInterval = 0.1f; // seconds
    bool isPaused = false;
    sf::Font font;
    sf::Text speedText;
    
public:
    GameOfLife(int w, int h) : width(w), height(h) {
        grid.resize(height, std::vector<bool>(width, false));
        nextGrid = grid;
        
        window.create(sf::VideoMode(width * cellSize, height * cellSize + 50), "Game of Life");
        
        // Initialize font and text
        font.loadFromFile("arial.ttf");
        speedText.setFont(font);
        speedText.setCharacterSize(20);
        speedText.setFillColor(sf::Color::White);
        speedText.setPosition(10, height * cellSize + 10);
        updateSpeedText();
        
        // Random initial state
        randomize();
    }
    
    void randomize() {
        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width; x++) {
                grid[y][x] = rand() % 2;
            }
        }
    }
    
    void updateSpeedText() {
        speedText.setString("Speed: " + std::to_string(1.0f/updateInterval) + " updates/sec");
    }
    
    int countNeighbors(int x, int y) {
        int count = 0;
        for(int dy = -1; dy <= 1; dy++) {
            for(int dx = -1; dx <= 1; dx++) {
                if(dx == 0 && dy == 0) continue;
                
                int nx = (x + dx + width) % width;
                int ny = (y + dy + height) % height;
                
                if(grid[ny][nx]) count++;
            }
        }
        return count;
    }
    
    void update() {
        if(isPaused) return;
        
        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width; x++) {
                int neighbors = countNeighbors(x, y);
                bool currentCell = grid[y][x];
                
                nextGrid[y][x] = (neighbors == 3) || (currentCell && neighbors == 2);
            }
        }
        
        grid.swap(nextGrid);
    }
    
    void run() {
        auto lastUpdate = std::chrono::high_resolution_clock::now();
        
        while(window.isOpen()) {
            sf::Event event;
            while(window.pollEvent(event)) {
                handleEvent(event);
            }
            
            auto currentTime = std::chrono::high_resolution_clock::now();
            float deltaTime = std::chrono::duration<float>(currentTime - lastUpdate).count();
            
            if(deltaTime >= updateInterval) {
                update();
                lastUpdate = currentTime;
            }
            
            draw();
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    
    void handleEvent(const sf::Event& event) {
        if(event.type == sf::Event::Closed)
            window.close();
            
        else if(event.type == sf::Event::KeyPressed) {
            switch(event.key.code) {
                case sf::Keyboard::Space:
                    isPaused = !isPaused;
                    break;
                case sf::Keyboard::R:
                    randomize();
                    break;
                case sf::Keyboard::Up:
                    updateInterval = std::max(0.01f, updateInterval - 0.05f);
                    updateSpeedText();
                    break;
                case sf::Keyboard::Down:
                    updateInterval = std::min(1.0f, updateInterval + 0.05f);
                    updateSpeedText();
                    break;
            }
        }
        else if(event.type == sf::Event::MouseButtonPressed) {
            int x = event.mouseButton.x / cellSize;
            int y = event.mouseButton.y / cellSize;
            if(y < height) {
                grid[y][x] = !grid[y][x];
            }
        }
    }
    
    void draw() {
        window.clear(sf::Color::Black);
        
        sf::RectangleShape cell(sf::Vector2f(cellSize-1, cellSize-1));
        cell.setFillColor(sf::Color::White);
        
        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width; x++) {
                if(grid[y][x]) {
                    cell.setPosition(x * cellSize, y * cellSize);
                    window.draw(cell);
                }
            }
        }
        
        window.draw(speedText);
        window.display();
    }
};

int main() {
    GameOfLife game(200, 200);  // 80x60 grid
    game.run();
    return 0;
}