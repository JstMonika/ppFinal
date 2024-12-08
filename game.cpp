#include <SFML/Graphics.hpp>
#include <X11/Xlib.h>

#include <iostream>
#include <vector>
#include <chrono>
#include <thread>

class GameOfLife {
private:
    sf::RenderWindow window;
    float cellSize = 3.0f;
    bool windowIsOpen = true;
    bool isPaused = false;
    sf::Font font;
    sf::Text speedTextCPU;
    sf::Text speedTextGPU;
    sf::Clock cpuClock, gpuClock;
    int cpuFrameTime = 0, gpuFrameTime = 0;

    int width, height;
    std::vector<std::vector<bool>> gridCPU;  // CPU grid
    std::vector<std::vector<bool>> nextGridCPU;
    std::vector<std::vector<bool>> gridGPU;  // GPU grid
    std::vector<std::vector<bool>> nextGridGPU;

    pthread_mutex_t mutex;

    // Calculating Time
    int updateCount = 0;
    float intervalDuration = 1.0f; // Default interval duration in seconds
    
public:
    GameOfLife(int w, int h) : width(w), height(h) {
        // 初始化兩個 grid
        gridCPU.resize(height, std::vector<bool>(width, false));
        nextGridCPU = gridCPU;
        gridGPU = gridCPU;
        nextGridGPU = gridCPU;
        
        // 創建雙倍寬度的窗口
        window.create(sf::VideoMode(width * cellSize * 2, height * cellSize + 50), "Game of Life - CPU vs GPU");
        
        // Initialize font and text
        if (!font.loadFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")) {
            std::cerr << "無法載入字體" << std::endl;
        }
        
        // CPU 文字
        speedTextCPU.setFont(font);
        speedTextCPU.setCharacterSize(20);
        speedTextCPU.setFillColor(sf::Color::White);
        speedTextCPU.setPosition(10, height * cellSize + 10);
        
        // GPU 文字
        speedTextGPU.setFont(font);
        speedTextGPU.setCharacterSize(20);
        speedTextGPU.setFillColor(sf::Color::White);
        speedTextGPU.setPosition(width * cellSize + 10, height * cellSize + 10);
        
        pthread_mutex_init(&mutex, NULL);

        randomize();
    }
    
    void randomize() {
        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width; x++) {
                int Rvalue = rand() % 3;
                bool value = Rvalue == 0;

                gridCPU[y][x] = value;
                gridGPU[y][x] = value;
            }
        }
    }

    int countNeighbors(const std::vector<std::vector<bool>>& grid, int x, int y) {
        int count = 0;
        for(int dy = -1; dy <= 1; dy++) {
            for(int dx = -1; dx <= 1; dx++) {
                if(dx == 0 && dy == 0) continue;
                
                int nx = x + dx;
                int ny = y + dy;
                if (ny < 0 || ny >= height) continue;
                if (nx < 0 || nx >= width) continue;
                
                if(grid[ny][nx]) count++;
            }
        }
        return count;
    }

    void updateCPU() {
        if(isPaused) return;
        
        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width; x++) {
                int neighbors = countNeighbors(gridCPU, x, y);
                bool currentCell = gridCPU[y][x];
                nextGridCPU[y][x] = (neighbors == 3) || (currentCell && neighbors == 2);
            }
        }

        pthread_mutex_lock(&mutex);
        gridCPU.swap(nextGridCPU);
        pthread_mutex_unlock(&mutex);
    }
    
    void updateGPU() {
        if(isPaused) return;
        
        // TODO: 實作 CUDA kernel
        // 暫時複製 CPU 的邏輯
        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width; x++) {
                int neighbors = countNeighbors(gridGPU, x, y);
                bool currentCell = gridGPU[y][x];
                nextGridGPU[y][x] = (neighbors == 3) || (currentCell && neighbors == 2);
            }
        }

        pthread_mutex_lock(&mutex);
        gridGPU.swap(nextGridGPU);
        pthread_mutex_unlock(&mutex);
    }
   
    void updateSpeedText() {
        speedTextCPU.setString("CPU FPS: " + std::to_string(cpuFrameTime));
        speedTextGPU.setString("GPU FPS: " + std::to_string(gpuFrameTime));
    }

    static void* CPUThread(void* arg) {
        GameOfLife* game = static_cast<GameOfLife*>(arg);
        game->CPU();
        return NULL;
    }

    static void* GPUThread(void* arg) {
        GameOfLife* game = static_cast<GameOfLife*>(arg);
        game->GPU();
        return NULL;   
    }
    
    void run() {
        pthread_t cpuThread, gpuThread;

        pthread_create(&cpuThread, NULL, CPUThread, this);
        pthread_create(&gpuThread, NULL, GPUThread, this);

        while(window.isOpen()) {
            sf::Event event;
            while(window.pollEvent(event)) {
                handleEvent(event);
            }

            draw();
            
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        windowIsOpen = false;

        pthread_join(cpuThread, NULL);
        pthread_join(gpuThread, NULL);
    }

    void CPU() {
        auto lastUpdate = std::chrono::high_resolution_clock::now();
        
        while(windowIsOpen) {
            auto currentTime = std::chrono::high_resolution_clock::now();
            float deltaTime = std::chrono::duration<float>(currentTime - lastUpdate).count();

            auto updateStart = std::chrono::high_resolution_clock::now();
            updateCPU();
            auto updateEnd = std::chrono::high_resolution_clock::now();
            float updateDuration = std::chrono::duration<float>(updateEnd - updateStart).count();
            
            updateCount++;

            // get time per update
            
            if (deltaTime >= intervalDuration) {
                updateCount = 0;

                cpuFrameTime = updateCount;

                lastUpdate = currentTime;
            }
        }
    }

    void GPU() {

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
            }
        }

        // else if(event.type == sf::Event::MouseButtonPressed) {
        //     int x = event.mouseButton.x / cellSize;
        //     int y = event.mouseButton.y / cellSize;
        //     if(y < height) {
        //         // grid[y][x] = !grid[y][x];
        //     }
        // }
    }
    
    
    void draw() {
        pthread_mutex_lock(&mutex);
        updateSpeedText();

        window.clear(sf::Color::Black);
        
        // 畫出分隔線
        sf::RectangleShape separator(sf::Vector2f(2, height * cellSize + 50));
        separator.setFillColor(sf::Color::White);
        separator.setPosition(width * cellSize, 0);
        
        // 繪製 CPU side
        sf::RectangleShape cellCPU(sf::Vector2f(cellSize-1, cellSize-1));
        cellCPU.setFillColor(sf::Color::White);
        
        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width; x++) {
                if(gridCPU[y][x]) {
                    cellCPU.setPosition(x * cellSize, y * cellSize);
                    window.draw(cellCPU);
                }
            }
        }
        
        // 繪製 GPU side
        sf::RectangleShape cellGPU(sf::Vector2f(cellSize-1, cellSize-1));
        cellGPU.setFillColor(sf::Color::Yellow);  // 使用不同顏色區分
        
        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width; x++) {
                if(gridGPU[y][x]) {
                    cellGPU.setPosition(width * cellSize + x * cellSize, y * cellSize);
                    window.draw(cellGPU);
                }
            }
        }
        
        window.draw(separator);
        window.draw(speedTextCPU);
        window.draw(speedTextGPU);
        window.display();

        pthread_mutex_unlock(&mutex);
    }
};

int main() {
    XInitThreads();

    GameOfLife game(300, 300);  // 80x60 grid
    game.run();
    return 0;
}