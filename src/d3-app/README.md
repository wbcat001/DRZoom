# D3 App - HDBSCAN Cluster Explorer

D3.js-based interactive visualization tool for exploring HDBSCAN clustering results with dimensionality reduction embeddings.

## Quick Start

### 1. Install Dependencies

```bash
npm install
```

### 2. Start Backend (Python FastAPI)

```bash
# Windows
run_backend.bat

# Linux/Mac
./run_backend.sh

# Or manually
cd src/backend
uvicorn main_d3:app --reload --port 8000
```

### 3. Start Frontend (Vite Dev Server)

```bash
npm run dev
```

### 4. Open Browser

Navigate to `http://localhost:5173`

## Project Structure

```
d3-app/
├── src/
│   ├── backend/              # Python FastAPI backend
│   │   ├── main_d3.py       # API endpoints
│   │   └── services/        # Data processing
│   ├── api/                 # Frontend API client
│   ├── components/          # React components
│   ├── store/               # State management (Context API)
│   ├── types/               # TypeScript types
│   └── utils/               # Utility functions
├── .env.local               # Environment variables
└── DATA_CONFIGURATION.md    # Data setup guide
```

## Configuration

See [DATA_CONFIGURATION.md](DATA_CONFIGURATION.md) for data setup instructions.

---

## React + TypeScript + Vite

This template provides a minimal setup to get React working in Vite with HMR and some ESLint rules.

Currently, two official plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Babel](https://babeljs.io/) (or [oxc](https://oxc.rs) when used in [rolldown-vite](https://vite.dev/guide/rolldown)) for Fast Refresh
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/) for Fast Refresh

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend updating the configuration to enable type-aware lint rules:

```js
export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...

      // Remove tseslint.configs.recommended and replace with this
      tseslint.configs.recommendedTypeChecked,
      // Alternatively, use this for stricter rules
      tseslint.configs.strictTypeChecked,
      // Optionally, add this for stylistic rules
      tseslint.configs.stylisticTypeChecked,

      // Other configs...
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```

You can also install [eslint-plugin-react-x](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-x) and [eslint-plugin-react-dom](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-dom) for React-specific lint rules:

```js
// eslint.config.js
import reactX from 'eslint-plugin-react-x'
import reactDom from 'eslint-plugin-react-dom'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...
      // Enable lint rules for React
      reactX.configs['recommended-typescript'],
      // Enable lint rules for React DOM
      reactDom.configs.recommended,
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```
