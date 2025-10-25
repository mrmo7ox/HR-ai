FROM node:20 AS build

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm install

COPY . .
RUN npm run build

FROM node:20-slim

WORKDIR /app

COPY --from=build /app ./
RUN npm install --omit=dev

EXPOSE 3000

CMD ["node", "build/"]