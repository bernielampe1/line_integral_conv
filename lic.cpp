// main.cpp
// C++ OpenGL LIC - Convolutional Line Integral Convolution (fragment shader)
// Build (Linux):
//   g++ main.cpp -o lic -lGL -lglfw -lGLEW
//
// Notes:
//  * Generates a swirl vector field and a white noise texture.
//  * Performs LIC in the fragment shader with forward/backward streamline integration.
//  * Toggle params at the top of the file or wire to keys as desired.

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <vector>
#include <random>
#include <iostream>
#include <cmath>
#include <string>

// ---------- Tunables ----------
static const int TEX_W = 1024;
static const int TEX_H = 1024;
static const int NOISE_W = 1024;
static const int NOISE_H = 1024;

// LIC parameters (also mirrored to shader uniforms)
static int   gSamples = 40;       // total samples (both directions combined)
static float gStep    = 1.2f;     // step in texels per sample (arc-length step)
static int   gUseHann = 1;        // 1=Hann kernel, 0=box average

// ------------------------------

static void checkShader(GLuint sh, const char* label) {
    GLint ok = GL_FALSE;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        GLint len = 0; glGetShaderiv(sh, GL_INFO_LOG_LENGTH, &len);
        std::string log(len, '\0');
        glGetShaderInfoLog(sh, len, nullptr, log.data());
        std::cerr << "Shader compile error (" << label << "):\n" << log << "\n";
        std::exit(1);
    }
}

static void checkProgram(GLuint prog, const char* label) {
    GLint ok = GL_FALSE;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        GLint len = 0; glGetProgramiv(prog, GL_INFO_LOG_LENGTH, &len);
        std::string log(len, '\0');
        glGetProgramInfoLog(prog, len, nullptr, log.data());
        std::cerr << "Program link error (" << label << "):\n" << log << "\n";
        std::exit(1);
    }
}

static GLuint makeShader(GLenum type, const char* src, const char* label) {
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);
    checkShader(sh, label);
    return sh;
}

static GLuint makeProgram(const char* vs, const char* fs, const char* label) {
    GLuint v = makeShader(GL_VERTEX_SHADER, vs, "VS");
    GLuint f = makeShader(GL_FRAGMENT_SHADER, fs, "FS");
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    checkProgram(p, label);
    glDeleteShader(v);
    glDeleteShader(f);
    return p;
}

// Fullscreen quad (NDC position + UV)
static GLuint makeFullscreenVAO() {
    float verts[] = {
        // x, y,   u, v
        -1.0f, -1.0f, 0.0f, 0.0f,
         1.0f, -1.0f, 1.0f, 0.0f,
        -1.0f,  1.0f, 0.0f, 1.0f,
         1.0f,  1.0f, 1.0f, 1.0f
    };
    GLuint vao=0, vbo=0;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
      glBindBuffer(GL_ARRAY_BUFFER, vbo);
      glBufferData(GL_ARRAY_BUFFER, sizeof(verts), verts, GL_STATIC_DRAW);
      glEnableVertexAttribArray(0);
      glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)0);
      glEnableVertexAttribArray(1);
      glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4*sizeof(float), (void*)(2*sizeof(float)));
    glBindVertexArray(0);
    return vao;
}

// Generate white noise texture (R8, [0..255])
static GLuint makeNoiseTexture(int w, int h) {
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> dist(0, 255);
    std::vector<unsigned char> noise(w*h);
    for (int i=0;i<w*h;++i) noise[i] = (unsigned char)dist(rng);

    GLuint tex=0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, w, h, 0, GL_RED, GL_UNSIGNED_BYTE, noise.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);  // blur a bit
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);      // periodic to reduce seams
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    return tex;
}

// Generate a swirl vector field centered at (0.5, 0.5): v = k * [-y, x]
// Stores RG16F with components in [-1,1] range (not normalized to length 1 at storage time).
static GLuint makeVectorField(int w, int h) {
    std::vector<float> data(w*h*2, 0.0f);
    float cx = 0.5f, cy = 0.5f;
    float k = 1.0f; // angular strength
    for (int j=0;j<h;++j) {
        for (int i=0;i<w;++i) {
            float u = (i + 0.5f) / float(w);
            float v = (j + 0.5f) / float(h);
            float x = u - cx;
            float y = v - cy;
            // Swirl field (counter-clockwise): [-y, x]
            float vx = -y;
            float vy =  x;
            // optional scale by k and normalize to avoid huge steps
            float len = std::sqrt(vx*vx + vy*vy) + 1e-6f;
            vx = (vx / len) * k;
            vy = (vy / len) * k;

            data[2*(j*w+i) + 0] = vx; // R
            data[2*(j*w+i) + 1] = vy; // G
        }
    }

    GLuint tex=0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    // Internal format: 16-bit float per channel for precision, two channels (RG)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RG16F, w, h, 0, GL_RG, GL_FLOAT, data.data());
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); // we’ll sample often, keep stable
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);      // wrap to keep streamlines seamless
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    return tex;
}

// Vertex shader: pass through
static const char* kVS = R"(
#version 330 core
layout(location = 0) in vec2 inPos;
layout(location = 1) in vec2 inUV;
out vec2 vUV;
void main() {
    vUV = inUV;
    gl_Position = vec4(inPos, 0.0, 1.0);
}
)";

// Fragment shader: LIC integration
// - uVector: RG16F (vx,vy) in approximately [-1,1], sampled and normalized per step
// - uNoise: R8/normalized
// - uStepTex: step length in TEXTURE SPACE (texels); shader converts to UV by dividing by texture size
// - uSamples: total sample count (both dirs), half forward and half backward
// - uUseHann: 1=Hann kernel, 0=box
static const char* kFS = R"(
#version 330 core
in vec2 vUV;
out vec4 fragColor;

uniform sampler2D uNoise;
uniform sampler2D uVector;
uniform int   uSamples;
uniform float uStepTex;  // in texels
uniform int   uUseHann;

float hannWeight(float t) {
    // t in [0,1]
    return 0.5 * (1.0 - cos(2.0*3.14159265359*t));
}

vec2 fieldDir(vec2 uv) {
    // Read vector; normalize to unit direction
    vec2 v = texture(uVector, uv).rg;
    float L = length(v);
    if (L < 1e-6) return vec2(0.0);
    return v / L;
}

void main() {
    // Texture size (for converting texel steps to UV)
    vec2 texSize = vec2(textureSize(uNoise, 0));
    float du = uStepTex / texSize.x; // assume square pixels
    float dv = uStepTex / texSize.y;

    // Total samples -> split into forward/backward halves
    int halfN = max(1, uSamples / 2);

    // Accumulator
    float acc = 0.0;
    float wsum = 0.0;

    // Center sample (optional – many LICs include it)
    float w0 = (uUseHann==1) ? hannWeight(0.5) : 1.0;
    float c0 = texture(uNoise, vUV).r;
    acc  += w0 * c0;
    wsum += w0;

    // Integrate forward
    vec2 p = vUV;
    vec2 d = fieldDir(p);
    for (int i=1; i<=halfN; ++i) {
        // Simple 2nd-order (midpoint) step in texture UV
        vec2 d1 = d;
        vec2 mid = p + 0.5 * vec2(du*d1.x, dv*d1.y);
        vec2 d2 = fieldDir(mid);
        p += vec2(du*d2.x, dv*d2.y);

        float t = float(i) / float(halfN+1); // [0..1]
        float w = (uUseHann==1) ? hannWeight(t) : 1.0;
        float s = texture(uNoise, p).r;
        acc  += w * s;
        wsum += w;

        // Update direction for next step
        d = d2;
    }

    // Integrate backward
    p = vUV;
    d = fieldDir(p);
    for (int i=1; i<=halfN; ++i) {
        vec2 d1 = d;
        vec2 mid = p - 0.5 * vec2(du*d1.x, dv*d1.y);
        vec2 d2 = fieldDir(mid);
        p -= vec2(du*d2.x, dv*d2.y);

        float t = float(i) / float(halfN+1);
        float w = (uUseHann==1) ? hannWeight(t) : 1.0;
        float s = texture(uNoise, p).r;
        acc  += w * s;
        wsum += w;

        d = d2;
    }

    float lic = acc / max(1e-6, wsum);
    // Optional contrast shaping (gamma). Comment out or tweak as desired.
    float gamma = 0.9;
    lic = pow(lic, gamma);

    fragColor = vec4(lic, lic, lic, 1.0);
}
)";

int main() {
    // --- Init GLFW ---
    if (!glfwInit()) {
        std::cerr << "Failed to init GLFW\n"; return -1;
    }
    // Core profile 3.3 (Mac users: also set GLFW_OPENGL_FORWARD_COMPAT to GL_TRUE)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    GLFWwindow* win = glfwCreateWindow(1024, 1024, "LIC (Convolutional Line Integral Convolution)", nullptr, nullptr);
    if (!win) { std::cerr << "Failed to create window\n"; glfwTerminate(); return -1; }
    glfwMakeContextCurrent(win);
    glfwSwapInterval(1); // vsync

    // --- Init GLEW ---
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) { std::cerr << "Failed to init GLEW\n"; return -1; }

    // --- Resources ---
    GLuint vao = makeFullscreenVAO();
    GLuint prog = makeProgram(kVS, kFS, "lic");

    // Textures
    GLuint texNoise  = makeNoiseTexture(NOISE_W, NOISE_H);
    GLuint texVector = makeVectorField(TEX_W, TEX_H);

    // --- Uniform bindings ---
    glUseProgram(prog);
    glUniform1i(glGetUniformLocation(prog, "uNoise"),  0); // texture unit 0
    glUniform1i(glGetUniformLocation(prog, "uVector"), 1); // texture unit 1

    // Initial LIC params
    GLint locSamples = glGetUniformLocation(prog, "uSamples");
    GLint locStepTex = glGetUniformLocation(prog, "uStepTex");
    GLint locUseHann = glGetUniformLocation(prog, "uUseHann");
    glUniform1i(locSamples, gSamples);
    glUniform1f(locStepTex, gStep);
    glUniform1i(locUseHann, gUseHann);

    // --- Render loop ---
    while (!glfwWindowShouldClose(win)) {
        // Simple key controls (hold briefly). Feel free to add debouncing if you want precise increments.
        if (glfwGetKey(win, GLFW_KEY_UP) == GLFW_PRESS)   { gSamples = std::min(200, gSamples+1); glUniform1i(locSamples, gSamples); }
        if (glfwGetKey(win, GLFW_KEY_DOWN) == GLFW_PRESS) { gSamples = std::max(4,   gSamples-1); glUniform1i(locSamples, gSamples); }
        if (glfwGetKey(win, GLFW_KEY_RIGHT) == GLFW_PRESS){ gStep = std::min(6.0f, gStep + 0.02f); glUniform1f(locStepTex, gStep); }
        if (glfwGetKey(win, GLFW_KEY_LEFT) == GLFW_PRESS) { gStep = std::max(0.1f, gStep - 0.02f); glUniform1f(locStepTex, gStep); }
        if (glfwGetKey(win, GLFW_KEY_H) == GLFW_PRESS)    { gUseHann = 1; glUniform1i(locUseHann, gUseHann); }
        if (glfwGetKey(win, GLFW_KEY_B) == GLFW_PRESS)    { gUseHann = 0; glUniform1i(locUseHann, gUseHann); }
        if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(win, GLFW_TRUE);

        int w, h; glfwGetFramebufferSize(win, &w, &h);
        glViewport(0, 0, w, h);
        glClearColor(0.02f, 0.02f, 0.025f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(prog);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texNoise);
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, texVector);

        glBindVertexArray(vao);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

        glfwSwapBuffers(win);
        glfwPollEvents();
    }

    glDeleteTextures(1, &texNoise);
    glDeleteTextures(1, &texVector);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(prog);
    glfwTerminate();
    return 0;
}
