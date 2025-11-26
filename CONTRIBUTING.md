# Camino hacia RAG\_dev v2.0.0

&#x20;Este proyecto busca construir un entorno colaborativo sobre el desarrollo de un sistema RAG.&#x20;

## 🚀 Flujo recomendado: Fork → Branch → Pull Request

Todas las contribuciones deben realizarse mediante el siguiente proceso:

### 1. Realiza un Fork del repositorio

En GitHub, haz clic en **Fork** para crear una copia del proyecto en tu cuenta.

### 2. Clona tu Fork

```bash
git clone https://github.com/<tu-usuario>/RAG_dev.git
cd RAG_dev
```

### 3. Crea una rama nueva para tu contribución

Utiliza un nombre descriptivo para la rama.

```bash
git checkout -b feature/nombre-de-la-mejora
```

Ejemplos:

- `feature/vectorstore-improvements`
- `fix/readme-format`
- `docs/add-usage-example`

### 4. Realiza tus cambios

Asegúrate de seguir la estructura del proyecto y buenas prácticas de código.

### 5. Confirma tus cambios

```bash
git add .
git commit -m "Descripción clara de la contribución"
```

### 6. Envía tu rama al repositorio remoto (tu Fork)

```bash
git push origin feature/nombre-de-la-mejora
```

### 7. Crea un Pull Request

En GitHub aparecerá un botón para iniciar un **Pull Request (PR)** hacia el repositorio original `erikycd/RAG_dev`, rama `main`.

## 📦 Base de contribución: Release estable

Por favor **basa tu trabajo en el último release estable**, actualmente:

```
v1.0.0
```

Puedes revisar el contenido del release para asegurar compatibilidad.

## 🧭 Lineamientos generales

### ✔ Estilo de código

- Mantén consistencia en nombres de archivos y funciones.
- Usa tipos y documentación cuando sea relevante.
- Para notebooks, incluye comentarios claros sobre propósito y entradas/salidas.

### ✔ Estructura de directorios

- `src/` → código fuente del pipeline RAG
- `notebooks/` → prototipos, experimentos y pruebas
- `data/` → datos de ejemplo o estructura esperada
- `docs/` → documentación adicional

### ✔ Cambios grandes

Si propones una modificación importante (reestructuración, nuevas dependencias, etc.), abre primero un **Issue** para discutirlo.

### ✔ Evita incluir

- Datos sensibles, privados o llaves
- Archivos innecesarios como checkpoints pesados o `.pyc`

## 📝 Revisión de Pull Requests

Todos los PRs serán revisados por el admin. del proyecto. La revisión puede incluir:

- Solicitudes de ajustes
- Comentarios sobre estructura o legibilidad
- Confirmación de compatibilidad con `main`

Las fusiones se realizarán usando **Squash and Merge** para mantener un historial limpio.
