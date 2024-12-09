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
const int MAX_UPDATES = 2000;

__constant__ int d_width, d_height, d_draw_frequency;

__global__ void kernel_update(bool* grid, bool* nextGrid) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    int idx = (y + 1) * (d_width + 2) + (x + 1);
    int neighbors, dx, dy, nx, ny;
    bool currentCell;

    neighbors = 0;

    for (dy = -1; dy <= 1; dy++) {
        for (dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;

            nx = x + dx;
            ny = y + dy;

            if (grid[(ny + 1) * (d_width + 2) + (nx + 1)]) {
                neighbors++;
            }
        }
    }

    currentCell = grid[idx];
    nextGrid[idx] = (neighbors == 3) || (currentCell && neighbors == 2);
}

class GameOfLife {
private:
    sf::RenderWindow window;
    float cellSize = 8.0f;
    bool windowIsOpen = true;
    bool isPaused = true;
    sf::Font font;
    sf::Text speedTextCPU;
    sf::Text speedTextGPU;
    sf::Text speedTextCPUPP;
    sf::Clock cpuClock, gpuClock, cpuppClock;
    float cpuFPS = 0, gpuFPS = 0, cpuppFPS = 0;

    int displaySize;
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
    int totalUpdateCountCPUPP = 0; // Total updates for CPU Parallel
    int totalUpdateCountGPU = 0; // Total updates for GPU

    bool drawCPU = false;
    bool drawCPUPP = false;
    bool drawGPU = false;

    bool doneCPU = false;
    bool doneCPUPP = false;
    bool doneGPU = false;

public:
    GameOfLife(int w, int h) : width(w), height(h) {

        displaySize = min(80, min(width, height));

        cHeight = (height + 31) / 32 * 32;
        cWidth = (width + 31) / 32 * 32;

        cudaMemcpyToSymbol(d_width, &cWidth, sizeof(int));
        cudaMemcpyToSymbol(d_height, &cHeight, sizeof(int));
        cudaMemcpyToSymbol(d_draw_frequency, &DRAW_FREQUENCY, sizeof(int));

        gridCPU.resize(height, std::vector<bool>(width, false));
        nextGridCPU = gridCPU;
        gridCPUPP = gridCPU;
        nextGridCPUPP = gridCPU;

        gridGPU = new bool[(cHeight + 2) * (cWidth + 2)];
        std::fill_n(gridGPU, (cHeight + 2) * (cWidth + 2), false);

        cudaMallocHost(&drawGridGPU, (cHeight + 2) * (cWidth + 2) * sizeof(bool));

        window.create(sf::VideoMode(displaySize * cellSize * 3 + 4, displaySize * cellSize + 70), "Game of Life - CPU vs GPU vs CPUPP");
        
        if (!font.loadFromFile("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")) {
            std::cerr << "無法載入字體" << std::endl;
        }
        
        speedTextCPU.setFont(font);
        speedTextCPU.setCharacterSize(20);
        speedTextCPU.setFillColor(sf::Color::White);
        speedTextCPU.setPosition(10, displaySize * cellSize + 10);

        speedTextCPUPP.setFont(font);
        speedTextCPUPP.setCharacterSize(20);
        speedTextCPUPP.setFillColor(sf::Color::White);
        speedTextCPUPP.setPosition(displaySize * cellSize + 10, displaySize * cellSize + 10);
        
        speedTextGPU.setFont(font);
        speedTextGPU.setCharacterSize(20);
        speedTextGPU.setFillColor(sf::Color::White);
        speedTextGPU.setPosition(displaySize * 2 * cellSize + 10, displaySize * cellSize + 10);
        
        randomize();

        pthread_mutex_init(&mutex, NULL);
    }
    
    void randomize() {
        for(int y = 0; y < height; y++) {
            for(int x = 0; x < width; x++) {
                int Rvalue = rand() % 3;
                bool value = Rvalue == 0;

                gridCPU[y][x] = value;
                gridCPUPP[y][x] = value;

                int idx = (y+1) * (cWidth + 2) + (x+1);
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
        #pragma omp parallel for collapse(2)
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
        float remainingCPU = MAX_UPDATES - totalUpdateCountCPU;
        float remainingCPUPP = MAX_UPDATES - totalUpdateCountCPUPP;
        float remainingGPU = MAX_UPDATES - totalUpdateCountGPU;
        
        if (doneCPU) {
            speedTextCPU.setString("CPU FPS: " + std::to_string(cpuFPS) + " (Done)\n");
        } else {
            float estTimeCPU = remainingCPU / cpuFPS;
            speedTextCPU.setString("CPU FPS: " + std::to_string(cpuFPS) + "\n" +
                                "Est. Time: " + std::to_string(estTimeCPU) + "s");
        }

        if (doneGPU) {
            speedTextGPU.setString("GPU FPS: " + std::to_string(gpuFPS) + " (Done)\n");
        } else {
            float estTimeGPU = remainingGPU / gpuFPS;
            speedTextGPU.setString("GPU FPS: " + std::to_string(gpuFPS) + "\n" +
                                "Est. Time: " + std::to_string(estTimeGPU) + "s");
        }

        if (doneCPUPP) {
            speedTextCPUPP.setString("CPU Parallel FPS: " + std::to_string(cpuppFPS) + " (Done)\n");
        } else {
            float estTimeCPUPP = remainingCPUPP / cpuppFPS;
            speedTextCPUPP.setString("CPU Parallel FPS: " + std::to_string(cpuppFPS) + "\n" +
                                    "Est. Time: " + std::to_string(estTimeCPUPP) + "s");
        }
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
            if (isPaused) continue;

            auto currentTime = std::chrono::high_resolution_clock::now();
            float deltaTime = std::chrono::duration<float>(currentTime - lastUpdate).count();

            updateCPU();
            
            totalUpdateCountCPU++;

            if (totalUpdateCountCPU >= MAX_UPDATES) {
                break;
            }

            if (totalUpdateCountCPU % DRAW_FREQUENCY == 0) {
                pthread_mutex_lock(&mutex);
                drawCPU = true;
                drawGridCPU = gridCPU;
                pthread_mutex_unlock(&mutex);
            }
            
            cpuFPS = totalUpdateCountCPU / deltaTime;
        }

        pthread_mutex_lock(&mutex);
        drawCPU = true;
        drawGridCPU = gridCPU;
        pthread_mutex_unlock(&mutex);

        doneCPU = true;
    }

    void GPU() {
        auto lastUpdate = std::chrono::high_resolution_clock::now();
        
        bool* d_grid, *d_nextGrid;
        cudaMalloc(&d_grid, (cWidth + 2) * (cHeight + 2) * sizeof(bool));
        cudaMemcpy(d_grid, gridGPU, (cWidth + 2) * (cHeight + 2) * sizeof(bool), cudaMemcpyHostToDevice);
        
        cudaMalloc(&d_nextGrid, (cWidth + 2) * (cHeight + 2) * sizeof(bool));
        cudaMemset(d_nextGrid, 0, (cWidth + 2) * (cHeight + 2) * sizeof(bool));

        dim3 block(32, 32);
        dim3 grid(cWidth / 32, cHeight / 32);

        cudaEvent_t prev;
        cudaEventCreate(&prev);

        while(windowIsOpen) {
            if (isPaused) continue;
            
            auto currentTime = std::chrono::high_resolution_clock::now();
            float deltaTime = std::chrono::duration<float>(currentTime - lastUpdate).count();

            
            kernel_update<<<grid, block>>>(d_grid, d_nextGrid);
            cudaEventRecord(prev);
            cudaEventSynchronize(prev);

            cudaMemcpy(d_grid, d_nextGrid, (cWidth + 2) * (cHeight + 2) * sizeof(bool), cudaMemcpyDeviceToDevice);
            
            totalUpdateCountGPU++;
            
            if (totalUpdateCountGPU >= MAX_UPDATES) {
                break;
            }

            if (totalUpdateCountGPU % DRAW_FREQUENCY == 0) {   
                pthread_mutex_lock(&mutex);
                drawGPU = true;
                cudaMemcpy(drawGridGPU, d_grid, (cWidth + 2) * (cHeight + 2) * sizeof(bool), cudaMemcpyDeviceToHost);
                pthread_mutex_unlock(&mutex);
            }
            
            gpuFPS = totalUpdateCountGPU / deltaTime;
        } 

        pthread_mutex_lock(&mutex);
        drawGPU = true;
        cudaMemcpy(drawGridGPU, d_grid, (cWidth + 2) * (cHeight + 2) * sizeof(bool), cudaMemcpyDeviceToHost);
        pthread_mutex_unlock(&mutex);

        doneGPU = true;
    }

    void CPUParallel() {
        auto lastUpdate = std::chrono::high_resolution_clock::now();
        
        while(windowIsOpen) {
            if (isPaused) continue;

            auto currentTime = std::chrono::high_resolution_clock::now();
            float deltaTime = std::chrono::duration<float>(currentTime - lastUpdate).count();

            updateCPUParallel();

            totalUpdateCountCPUPP++;
            
            if (totalUpdateCountCPUPP >= MAX_UPDATES) {
                break;
            }

            if (totalUpdateCountCPUPP % DRAW_FREQUENCY == 0) {
                pthread_mutex_lock(&mutex);
                drawCPUPP = true;
                drawGridCPUPP = gridCPUPP;
                pthread_mutex_unlock(&mutex);
            }
            
            cpuppFPS = totalUpdateCountCPUPP / deltaTime;
        }

        pthread_mutex_lock(&mutex);
        drawCPUPP = true;
        drawGridCPUPP = gridCPUPP;
        pthread_mutex_unlock(&mutex);

        doneCPUPP = true;
    }
    
    void handleEvent(const sf::Event& event) {
        if(event.type == sf::Event::Closed)
            window.close();
            
        else if(event.type == sf::Event::KeyPressed) {
            switch(event.key.code) {
                case sf::Keyboard::Space:
                        isPaused = !isPaused;
                    break;
                default:
                    break;
            }
        }
    }
    
    void draw() {
        pthread_mutex_lock(&mutex);

        updateSpeedText();

        window.clear(sf::Color::Black);
        
        sf::RectangleShape separator(sf::Vector2f(2, displaySize * cellSize + 50));
        separator.setFillColor(sf::Color::White);
        separator.setPosition(displaySize * cellSize, 0);
        
        sf::RectangleShape separator2(sf::Vector2f(2, displaySize * cellSize + 50));
        separator2.setFillColor(sf::Color::White);
        separator2.setPosition(displaySize * 2 * cellSize + 2, 0);

        sf::RectangleShape cellCPU(sf::Vector2f(cellSize-1, cellSize-1));
        cellCPU.setFillColor(sf::Color::White);
        
        for(int y = 0; y < displaySize; y++) {
            for(int x = 0; x < displaySize; x++) {
                if(drawGridCPU[y][x]) {
                    cellCPU.setPosition(x * cellSize, y * cellSize);
                    window.draw(cellCPU);
                }
            }
        }
        
        sf::RectangleShape cellCPUPP(sf::Vector2f(cellSize-1, cellSize-1));
        cellCPUPP.setFillColor(sf::Color::Green);

        for(int y = 0; y < displaySize; y++) {
            for(int x = 0; x < displaySize; x++) {
                if(drawGridCPUPP[y][x]) {
                    cellCPUPP.setPosition(displaySize * cellSize + x * cellSize + 2, y * cellSize);
                    window.draw(cellCPUPP);
                }
            }
        }

        sf::RectangleShape cellGPU(sf::Vector2f(cellSize-1, cellSize-1));
        cellGPU.setFillColor(sf::Color::Yellow);
        
        for(int y = 0; y < displaySize; y++) {
            for(int x = 0; x < displaySize; x++) {
                int idx = (y + 1) * (cWidth + 2) + (x + 1);

                if(drawGridGPU[idx]) {
                    cellGPU.setPosition(displaySize * 2 * cellSize + x * cellSize + 4, y * cellSize);
                    window.draw(cellGPU);
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

    GameOfLife game(1024, 1024);
    game.run();
    return 0;
}