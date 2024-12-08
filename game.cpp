#include <SFML/Graphics.hpp>
#include <X11/Xlib.h>

#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <omp.h>


class GameOfLife {
private:
    sf::RenderWindow window;
    float cellSize = 3.0f;
    bool windowIsOpen = true;
    bool isPaused = false;
    sf::Font font;
    sf::Text speedTextCPU;
    sf::Text speedTextGPU;
    sf::Text speedTextCPUPP;
    sf::Clock cpuClock, gpuClock, cpuppClock;
    int cpuFrameTime = 0, gpuFrameTime = 0, cpuPPFrameTime = 0;

    int width, height;
    std::vector<std::vector<bool>> gridCPU;  // CPU grid
    std::vector<std::vector<bool>> nextGridCPU;
    std::vector<std::vector<bool>> gridGPU;  // GPU grid
    std::vector<std::vector<bool>> nextGridGPU;
    std::vector<std::vector<bool>> gridCPUPP;  // CPUPP grid
    std::vector<std::vector<bool>> nextGridCPUPP;


    pthread_mutex_t mutex;

    // Calculating Time
    int updateCountCPU = 0;
    int updateCountCPUPP = 0;
    int updateCountGPU = 0;
    int totalUpdateCountCPU = 0; // Total updates for CPU
    int totalUpdateCountCPUPP = 0; // Total updates for GPU
    int totalUpdateCountGPU = 0; // Total updates for GPU
    int maxUpdates = 300;


    float intervalDuration = 1.0f; // Default interval duration in seconds
    
public:
    GameOfLife(int w, int h) : width(w), height(h) {
        // 初始化兩個 grid
        gridCPU.resize(height, std::vector<bool>(width, false));
        nextGridCPU = gridCPU;
        gridGPU = gridCPU;
        nextGridGPU = gridCPU;
        gridCPUPP = gridCPU;
        nextGridCPUPP = gridCPU;

        // 創建三倍寬度的窗口
        window.create(sf::VideoMode(width * cellSize * 3, height * cellSize + 50), "Game of Life - CPU vs GPU vs CPUPP");
        
        // Initialize font and text
        if (!font.loadFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")) {
            std::cerr << "無法載入字體" << std::endl;
        }
        
        // CPU 文字
        speedTextCPU.setFont(font);
        speedTextCPU.setCharacterSize(20);
        speedTextCPU.setFillColor(sf::Color::White);
        speedTextCPU.setPosition(10, height * cellSize + 10);
        
        speedTextCPUPP.setFont(font);
        speedTextCPUPP.setCharacterSize(20);
        speedTextCPUPP.setFillColor(sf::Color::White);
        speedTextCPUPP.setPosition(width * 2 * cellSize + 10, height * cellSize + 10);

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
                gridCPUPP[y][x] = value;

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

    void updateCPUParallel() {
        if (isPaused) return;

        #pragma omp parallel for collapse(2)
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int neighbors = countNeighbors(gridCPUPP, x, y);
                bool currentCell = gridCPUPP[y][x];
                nextGridCPU[y][x] = (neighbors == 3) || (currentCell && neighbors == 2);
            }
        }

        pthread_mutex_lock(&mutex);
        gridCPUPP.swap(nextGridCPUPP);
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
        speedTextCPUPP.setString("CPUPP FPS: " + std::to_string(cpuPPFrameTime));
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
            
            updateCountCPUCPU++;
            totalUpdateCountCPU++; // Increment total CPU updates
            std::cout << totalUpdateCountCPU << std::endl;
            // get time per update
            
            if (deltaTime >= intervalDuration) {
                cpuFrameTime = updateCountCPU;
                updateCountCPU = 0;
                lastUpdate = currentTime;
            }
        }
    }

    void GPU() {
        
    }

    void CPUParallel() {
        auto lastUpdate = std::chrono::high_resolution_clock::now();
        
        while(windowIsOpen) {
            auto currentTime = std::chrono::high_resolution_clock::now();
            float deltaTime = std::chrono::duration<float>(currentTime - lastUpdate).count();

            auto updateStart = std::chrono::high_resolution_clock::now();
            updateCPUParallel();
            auto updateEnd = std::chrono::high_resolution_clock::now();
            float updateDuration = std::chrono::duration<float>(updateEnd - updateStart).count();
            
            updateCountCPUPP++;

            // get time per update
            
            if (deltaTime >= intervalDuration) {

                cpuPPFrameTime = updateCountCPUPP;
                updateCountCPUPP = 0;
                lastUpdate = currentTime;
            }
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
                default:
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
        
        sf::RectangleShape separator2(sf::Vector2f(2, height * cellSize + 50));
        separator2.setFillColor(sf::Color::White);
        separator2.setPosition(width * 2 * cellSize, 0);

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
        
        // 繪製 CPUPP side
        sf::RectangleShape cellCPUPP(sf::Vector2f(cellSize-1, cellSize-1));
        cellCPUPP.setFillColor(sf::Color::Green);  // 使用不同顏色區分

        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width; x++) {
                if(gridCPUPP[y][x]) {
                    cellCPUPP.setPosition(width * 2 * cellSize + x * cellSize, y * cellSize);
                    window.draw(cellCPUPP);
                }
            }
        }

        window.draw(separator);
        window.draw(separator2);
        window.draw(speedTextCPU);
        window.draw(speedTextGPU);
        window.draw(speedTextCPUPP);
        window.display();

        pthread_mutex_unlock(&mutex);
    }
};

int main() {
    XInitThreads();

    GameOfLife game(100, 100);  // 80x60 grid
    game.run();
    return 0;
}