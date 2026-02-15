# Build stage
FROM node:20-slim AS build

ARG VITE_API_BASE_URL=http://localhost:8000
ENV VITE_API_BASE_URL=${VITE_API_BASE_URL}

WORKDIR /app

COPY package.json package-lock.json* .npmrc* ./
RUN npm install

COPY . ./
RUN npm run build

# Runtime stage using nginx
FROM nginx:1.27-alpine AS runtime

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
