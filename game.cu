#include <SFML/Graphics.hpp>
#include <X11/Xlib.h>

#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <omp.h>
#include <sched.h>
#include <pthread.h>

class GameOfLife {
private:
    sf::RenderWindow window;
    float cellSize = 3.0f;
    bool windowIsOpen = true;
    sf::Font font;
    sf::Text speedTextCPU;
    sf::Text speedTextGPU;
    sf::Text speedTextCPUPP;
    sf::Clock cpuClock, gpuClock, cpuppClock;
    float cpuFrameTime = 0, gpuFrameTime = 0, cpuPPFrameTime = 0;

    int width, height;
    std::vector<std::vector<bool>> gridCPU;  // CPU grid
    std::vector<std::vector<bool>> nextGridCPU;
    std::vector<std::vector<bool>> gridGPU;  // GPU grid
    std::vector<std::vector<bool>> nextGridGPU;
    std::vector<std::vector<bool>> gridCPUPP;  // CPUPP grid
    std::vector<std::vector<bool>> nextGridCPUPP;

    std::vector<std::vector<bool>> drawGridCPU;
    std::vector<std::vector<bool>> drawGridCPUPP;
    std::vector<std::vector<bool>> drawGridGPU;

    pthread_mutex_t mutex;

    // Calculating Time
    int updateCountCPU = 0;
    int updateCountCPUPP = 0;
    int updateCountGPU = 0;
    int totalUpdateCountCPU = 0; // Total updates for CPU
    int totalUpdateCountCPUPP = 0; // Total updates for GPU
    int totalUpdateCountGPU = 0; // Total updates for GPU

    bool drawCPU = false;
    bool drawCPUPP = false;
    bool drawGPU = false;

    int maxUpdates = std::numeric_limits<int>::max(); // Maximum updates

public:
    GameOfLife(int w, int h) : width(w), height(h) {
        // 初始化兩個 grid
        gridCPU.resize(height, std::vector<bool>(width, false));
        nextGridCPU = gridCPU;
        gridCPUPP = gridCPU;
        nextGridCPUPP = gridCPU;

        gridGPU.resize(height + 2, std::vector<bool>(width + 2, false));
        nextGridGPU = gridGPU;

        // 創建三倍寬度的窗口
        window.create(sf::VideoMode(150 * cellSize * 3, 150 * cellSize + 50), "Game of Life - CPU vs GPU vs CPUPP");
        
        // Initialize font and text
        if (!font.loadFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")) {
            std::cerr << "無法載入字體" << std::endl;
        }
        
        // CPU 文字
        speedTextCPU.setFont(font);
        speedTextCPU.setCharacterSize(20);
        speedTextCPU.setFillColor(sf::Color::White);
        speedTextCPU.setPosition(10, 150 * cellSize + 10);

        speedTextCPUPP.setFont(font);
        speedTextCPUPP.setCharacterSize(20);
        speedTextCPUPP.setFillColor(sf::Color::White);
        speedTextCPUPP.setPosition(150 * cellSize + 10, 150 * cellSize + 10);
        
        // GPU 文字
        speedTextGPU.setFont(font);
        speedTextGPU.setCharacterSize(20);
        speedTextGPU.setFillColor(sf::Color::White);
        speedTextGPU.setPosition(150 * 2 * cellSize + 10, 150 * cellSize + 10);
        
        randomize();

        pthread_mutex_init(&mutex, NULL);
    }
    
    void randomize() {
        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width; x++) {
                int Rvalue = rand() % 2;
                bool value = Rvalue == 0;

                gridCPU[y][x] = value;
                gridCPUPP[y][x] = value;

                gridGPU[y+1][x+1] = value;
            }
        }
        
        drawGridCPU = gridCPU;
        drawGridCPUPP = gridCPU;
        drawGridGPU = gridCPU;
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
        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width; x++) {
                int neighbors = countNeighbors(gridCPU, x, y);
                bool currentCell = gridCPU[y][x];
                nextGridCPU[y][x] = (neighbors == 3) || (currentCell && neighbors == 2);
            }
        }

        gridCPU.swap(nextGridCPU);
    }

    void updateCPUParallel() {
        #pragma omp parallel for collapse(2) schedule(dynamic, 64)
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int neighbors = countNeighbors(gridCPUPP, x, y);
                bool currentCell = gridCPUPP[y][x];
                nextGridCPUPP[y][x] = (neighbors == 3) || (currentCell && neighbors == 2);
            }
        }

        gridCPUPP.swap(nextGridCPUPP);
        
    }

    void updateGPU() {
        // TODO: 實作 CUDA kernel
        // 暫時複製 CPU 的邏輯
        for(int y = 1; y <= height; y++) {
            for(int x = 1; x <= width; x++) {
                int neighbors = countNeighbors(gridGPU, x, y);
                bool currentCell = gridGPU[y][x];
                nextGridGPU[y][x] = (neighbors == 3) || (currentCell && neighbors == 2);
            }
        }

        gridGPU.swap(nextGridGPU);
    }
   
    void updateSpeedText() {
        speedTextCPU.setString("CPU FPS: " + std::to_string(cpuFrameTime));
        speedTextGPU.setString("GPU FPS: " + std::to_string(gpuFrameTime));
        speedTextCPUPP.setString("CPU Parallel FPS: " + std::to_string(cpuPPFrameTime));
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
    
    static void* CPUPPThread(void* arg) {
        GameOfLife* game = static_cast<GameOfLife*>(arg);
        game->CPUParallel();
        return NULL;
    }

    void run() {
        pthread_t cpuThread, cpuPPThread, gpuThread;

        pthread_create(&cpuThread, NULL, CPUThread, this);
        pthread_create(&gpuThread, NULL, GPUThread, this);
        pthread_create(&cpuPPThread, NULL, CPUPPThread, this);

        draw();

        auto lastUpdate = std::chrono::high_resolution_clock::now();

        while(window.isOpen()) {
            sf::Event event;
            while(window.pollEvent(event)) {
                handleEvent(event);
            }

            auto currentTime = std::chrono::high_resolution_clock::now();
            float deltaTime = std::chrono::duration<float>(currentTime - lastUpdate).count();

            if (drawCPU or drawCPUPP or drawGPU or deltaTime >= 1.0) {
                drawCPU = false;
                drawCPUPP = false;
                drawGPU = false;
                lastUpdate = currentTime;                

                draw();
            }
        }

        windowIsOpen = false;

        pthread_join(cpuThread, NULL);
        pthread_join(gpuThread, NULL);
        pthread_join(cpuPPThread, NULL);
    }

    void CPU() {
        auto lastUpdate = std::chrono::high_resolution_clock::now();
        
        while(windowIsOpen) {
            auto currentTime = std::chrono::high_resolution_clock::now();
            float deltaTime = std::chrono::duration<float>(currentTime - lastUpdate).count();

            if (totalUpdateCountCPU >= maxUpdates) {
                break;
            }

            updateCPU();
            
            totalUpdateCountCPU++; // Increment total CPU updates

            if (totalUpdateCountCPU % 10 == 0) {
                drawCPU = true;

                pthread_mutex_lock(&mutex);
                drawGridCPU = gridCPU;
                pthread_mutex_unlock(&mutex);
            }
            
            cpuFrameTime = totalUpdateCountCPU / deltaTime;
        }
    }

    void GPU() {
        
    }

    void CPUParallel() {
        auto lastUpdate = std::chrono::high_resolution_clock::now();
        
        while(windowIsOpen) {
            auto currentTime = std::chrono::high_resolution_clock::now();
            float deltaTime = std::chrono::duration<float>(currentTime - lastUpdate).count();

            if (totalUpdateCountCPUPP >= maxUpdates) {
                break;
            }

            updateCPUParallel();

            totalUpdateCountCPUPP++; // Increment total CPU updates

            if (totalUpdateCountCPUPP % 10 == 0) {
                drawCPUPP = true;

                pthread_mutex_lock(&mutex);
                drawGridCPUPP = gridCPUPP;
                pthread_mutex_unlock(&mutex);
            }
            
            cpuPPFrameTime = totalUpdateCountCPUPP / deltaTime;
        }
    }
    
    void handleEvent(const sf::Event& event) {
        if(event.type == sf::Event::Closed)
            window.close();
            
        else if(event.type == sf::Event::KeyPressed) {
            switch(event.key.code) {
                case sf::Keyboard::R:
                    randomize();
                    draw();
                    break;
                default:
                    break;
            }
        }
    }
    
    void draw() {
        updateSpeedText();

        window.clear(sf::Color::Black);
        
        // 畫出分隔線
        sf::RectangleShape separator(sf::Vector2f(2, 150 * cellSize + 50));
        separator.setFillColor(sf::Color::White);
        separator.setPosition(150 * cellSize, 0);
        
        sf::RectangleShape separator2(sf::Vector2f(2, 150 * cellSize + 50));
        separator2.setFillColor(sf::Color::White);
        separator2.setPosition(150 * 2 * cellSize, 0);

        pthread_mutex_lock(&mutex);

        // 繪製 CPU side
        sf::RectangleShape cellCPU(sf::Vector2f(cellSize-1, cellSize-1));
        cellCPU.setFillColor(sf::Color::White);
        
        for(int y = 0; y < 150; y++) {
            for(int x = 0; x < 150; x++) {
                if(drawGridCPU[y][x]) {
                    cellCPU.setPosition(x * cellSize, y * cellSize);
                    window.draw(cellCPU);
                }
            }
        }
        
        // 繪製 CPUPP side
        sf::RectangleShape cellCPUPP(sf::Vector2f(cellSize-1, cellSize-1));
        cellCPUPP.setFillColor(sf::Color::Green);  // 使用不同顏色區分

        for(int y = 0; y < 150; y++) {
            for(int x = 0; x < 150; x++) {
                if(drawGridCPUPP[y][x]) {
                    cellCPUPP.setPosition(150 * cellSize + x * cellSize, y * cellSize);
                    window.draw(cellCPUPP);
                }
            }
        }

        // 繪製 GPU side
        sf::RectangleShape cellGPU(sf::Vector2f(cellSize-1, cellSize-1));
        cellGPU.setFillColor(sf::Color::Yellow);  // 使用不同顏色區分
        
        for(int y = 1; y <= 150; y++) {
            for(int x = 1; x <= 150; x++) {
                if(drawGridGPU[y][x]) {
                    cellGPU.setPosition(150 * 2 * cellSize + x * cellSize, y * cellSize);
                    window.draw(cellGPU);
                }
            }
        }
        
        pthread_mutex_unlock(&mutex);

        window.draw(separator);
        window.draw(separator2);
        window.draw(speedTextCPU);
        window.draw(speedTextGPU);
        window.draw(speedTextCPUPP);
        window.display();
    }
};

int main() {
    XInitThreads();

    GameOfLife game(1000, 1000);  // 80x60 grid
    game.run();
    return 0;
}