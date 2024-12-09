#include <SFML/Graphics.hpp>
#include <X11/Xlib.h>

#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <omp.h>
#include <sched.h>
#include <pthread.h>


const int DRAW_FREQUENCY = 50;
const int MAX_UPDATES = 500000;

__constant__ int d_width, d_height, d_draw_frequency;

__global__ void kernel_update(bool* grid, bool* nextGrid, bool* drawGrid) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    nextGrid[y * (d_width + 2) + x] = false;
    nextGrid[(y + 2) * (d_width + 2) + x] = false;
    nextGrid[y * (d_width + 2) + x + 2] = false;
    nextGrid[(y + 2) * (d_width + 2) + x + 2] = false;
    
    int idx = (y + 1) * (d_width + 2) + (x + 1);
    int c, neighbors, dx, dy, nx, ny;
    bool currentCell;

    for (c = 0; c < d_draw_frequency; c++) {
        neighbors = 0;

        // unroll
        for (dy = -1; dy <= 1; dy++) {
            for (dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;

                nx = x + dx;
                ny = y + dy;

                if (grid[ny * (d_width + 2) + nx]) neighbors++;
            }
        }

        currentCell = grid[idx];
        nextGrid[idx] = (neighbors == 3) || (currentCell && neighbors == 2);

        // copy nextGrid to grid
        grid[idx] = nextGrid[idx];
    }

    drawGrid[idx] = grid[idx];
}

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
    float cpuFPS = 0, gpuFPS = 0, cpuppFPS = 0;

    int width, height;
    int cWidth, cHeight;

    std::vector<std::vector<bool>> gridCPU;  // CPU grid
    std::vector<std::vector<bool>> nextGridCPU;
    std::vector<std::vector<bool>> gridCPUPP;  // CPUPP grid
    std::vector<std::vector<bool>> nextGridCPUPP;
    std::vector<std::vector<bool>> drawGridCPU;
    std::vector<std::vector<bool>> drawGridCPUPP;
    
    bool* gridGPU;  // GPU grid
    bool* drawGridGPU;

    pthread_mutex_t mutex;

    // Calculating Time
    int totalUpdateCountCPU = 0; // Total updates for CPU
    int totalUpdateCountCPUPP = 0; // Total updates for GPU
    int totalUpdateCountGPU = 0; // Total updates for GPU

    bool drawCPU = false;
    bool drawCPUPP = false;
    bool drawGPU = false;

public:
    GameOfLife(int w, int h) : width(w), height(h) {

        cHeight = (height + 31) / 32 * 32;
        cWidth = (width + 31) / 32 * 32;

        cudaMemcpyToSymbol(d_width, &cWidth, sizeof(int));
        cudaMemcpyToSymbol(d_height, &cHeight, sizeof(int));
        cudaMemcpyToSymbol(d_draw_frequency, &DRAW_FREQUENCY, sizeof(int));

        // 初始化兩個 grid
        gridCPU.resize(height, std::vector<bool>(width, false));
        nextGridCPU = gridCPU;
        gridCPUPP = gridCPU;
        nextGridCPUPP = gridCPU;

        gridGPU = new bool[(cHeight + 2) * (cWidth + 2)]();
        cudaMallocHost(&drawGridGPU, (cHeight + 2) * (cWidth + 2) * sizeof(bool));

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

                int idx = (y+1) * (width + 2) + (x+1);
                gridGPU[idx] = value;
            }
        }
        
        drawGridCPU = gridCPU;
        drawGridCPUPP = gridCPU;
        drawGridGPU = gridGPU;
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
   
    void updateSpeedText() {
        speedTextCPU.setString("CPU FPS: " + std::to_string(cpuFPS));
        speedTextGPU.setString("GPU FPS: " + std::to_string(gpuFPS));
        speedTextCPUPP.setString("CPU Parallel FPS: " + std::to_string(cpuppFPS));
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

            if (totalUpdateCountCPU >= MAX_UPDATES) {
                break;
            }

            updateCPU();
            
            totalUpdateCountCPU++; // Increment total CPU updates

            if (totalUpdateCountCPU % DRAW_FREQUENCY == 0) {
                pthread_mutex_lock(&mutex);
                drawCPU = true;
                drawGridCPU = gridCPU;
                pthread_mutex_unlock(&mutex);
            }
            
            cpuFPS = totalUpdateCountCPU / deltaTime;
        }
    }

    void GPU() {
        auto lastUpdate = std::chrono::high_resolution_clock::now();
        
        bool* d_grid, *d_nextGrid, *d_drawGrid;
        cudaMalloc(&d_grid, (cWidth + 2) * (cHeight + 2) * sizeof(bool));
        cudaMemcpy(d_grid, gridGPU, (cWidth + 2) * (cHeight + 2) * sizeof(bool), cudaMemcpyHostToDevice);
        
        cudaMalloc(&d_nextGrid, (cWidth + 2) * (cHeight + 2) * sizeof(bool));
        cudaMalloc(&d_drawGrid, (cWidth + 2) * (cHeight + 2) * sizeof(bool));

        dim3 block(32, 32);
        dim3 grid(cWidth / 32, cHeight / 32);

        // cudaStream_t stream[3];
        // for (int i = 0; i < 3; i++) {
        //     cudaStreamCreate(&stream[i]);
        // }

        int now = 0;
        while(windowIsOpen) {
            auto currentTime = std::chrono::high_resolution_clock::now();
            float deltaTime = std::chrono::duration<float>(currentTime - lastUpdate).count();

            if (totalUpdateCountGPU >= MAX_UPDATES) {
                break;
            }

            kernel_update<<<grid, block>>>(d_grid, d_nextGrid, d_drawGrid);
            // kernel_update<<<grid, block, 0, stream[now]>>>(d_grid, d_nextGrid, d_drawGrid);
            
            totalUpdateCountGPU += DRAW_FREQUENCY; // Increment total CPU updates

            pthread_mutex_lock(&mutex);
            drawGPU = true;
            cudaMemcpy(drawGridGPU, d_drawGrid, (cWidth + 2) * (cHeight + 2) * sizeof(bool), cudaMemcpyDeviceToHost);
            // cudaMemcpyAsync(drawGridGPU, d_drawGrid, (cWidth + 2) * (cHeight + 2) * sizeof(bool), cudaMemcpyDeviceToHost, stream[now]);
            pthread_mutex_unlock(&mutex);
            
            now = (now + 1) % 3;
            gpuFPS = totalUpdateCountGPU / deltaTime;
        } 
    }

    void CPUParallel() {
        auto lastUpdate = std::chrono::high_resolution_clock::now();
        
        while(windowIsOpen) {
            auto currentTime = std::chrono::high_resolution_clock::now();
            float deltaTime = std::chrono::duration<float>(currentTime - lastUpdate).count();

            if (totalUpdateCountCPUPP >= MAX_UPDATES) {
                break;
            }

            updateCPUParallel();

            totalUpdateCountCPUPP++; // Increment total CPU updates

            if (totalUpdateCountCPUPP % DRAW_FREQUENCY == 0) {
                pthread_mutex_lock(&mutex);
                drawCPUPP = true;
                drawGridCPUPP = gridCPUPP;
                pthread_mutex_unlock(&mutex);
            }
            
            cpuppFPS = totalUpdateCountCPUPP / deltaTime;
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
        
        for(int y = 0; y < 150; y++) {
            for(int x = 0; x < 150; x++) {
                int idx = (y + 1) * (width + 2) + (x + 1);

                if(drawGridGPU[idx]) {
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