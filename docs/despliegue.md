# Despliegue del MVP en AWS (ticket 3.4)

Este documento describe cómo se despliega la API y el dashboard en una instancia EC2 de AWS Academy Learner Lab (región us-east-1), con Docker Compose. Es una demostración técnica: el modelo base tiene un desempeño cercano al azar y sus predicciones no deben usarse para decisiones clínicas.

## Reglas de confidencialidad

- Ningún dato a nivel de participante llega a GitHub, a una imagen Docker ni a la instancia. El archivo SPSS y cualquier derivado se quedan fuera de EC2.
- Solo se copian a la instancia los tres archivos del modelo (imputer_mice.pkl, baseline.pkl y baseline_meta.json). Antes de hacerlo hay que tener la confirmación de la Fundación Canguro de que un modelo entrenado puede salir de su entorno.
- Las pruebas de la demostración usan valores sintéticos o escritos a mano.
- El acceso es restringido: no es público. El grupo de seguridad solo permite las IP del equipo mientras dure la demostración, nunca 0.0.0.0/0, y la instancia se termina al final.

## Arquitectura

Dos contenedores en una instancia EC2 t3.small con Ubuntu Server 24.04 LTS y 20 GB de disco, definidos en deploy/docker-compose.prod.yml. El dashboard (puerto 8501) llama a la API por el nombre del servicio dentro de la red de Compose, y la API (puerto 8000) lee los modelos desde la carpeta modelos/ montada en solo lectura. Los modelos no van dentro de la imagen.

## 1. Preparar los modelos (terminal local)

Desde la raíz del repositorio, con acceso a los datos por DVC:

```
uv run dvc pull
uv run python -m src.pipeline --strict
uv run python -m src.models.train_baseline
ls -l modelos/
```

Deben existir imputer_mice.pkl, baseline.pkl y baseline_meta.json.

## 2. Iniciar el laboratorio y lanzar la instancia

1. En Canvas, Modules, abrir AWS Academy Learner Lab y pulsar Start Lab. Esperar a que el círculo junto a AWS se ponga verde y revisar el monto usado en la parte superior.
2. Abrir la consola de AWS (us-east-1). El usuario voclabs debe verse arriba a la derecha.
3. EC2, Launch instance, con estos valores: nombre proyectofinal-deploy, AMI Ubuntu Server 24.04 LTS, tipo t3.small, almacenamiento 20 GB, y un par de llaves RSA nuevo en formato .pem guardado en una carpeta local que esté en .gitignore.
4. Cuando la instancia esté en Running, anotar la IP pública y la IP privada.

## 3. Abrir los puertos del dashboard y la API

En EC2, Instances, pestaña Security, abrir el grupo de seguridad, Edit inbound rules, y agregar dos reglas Custom TCP: puerto 8501 y puerto 8000. En Source elegir My IP, y agregar las IP de los demás integrantes si van a probar. No usar 0.0.0.0/0. Las reglas se editan en Inbound, no en Outbound.

## 4. Conectar e instalar Docker

Terminal local:

```
chmod 400 <llave>.pem
ssh -i <llave>.pem ubuntu@<IP_PUBLICA>
```

Terminal de la instancia (SSH). El primer comando puede avisar que los paquetes no están instalados, lo cual es normal:

```
sudo apt-get remove docker docker-engine docker.io containerd runc docker-doc podman-docker
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg zip unzip -y
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y
sudo docker run hello-world
```

## 5. Clonar el repositorio (terminal de la instancia)

```
git clone https://github.com/MAIA-FinalProject/ProyectoFinal-FundacionCanguro.git
cd ProyectoFinal-FundacionCanguro
ls modelos/
```

La carpeta modelos/ existe y está vacía (solo contiene .gitkeep).

## 6. Copiar los modelos (terminal local)

Desde la raíz del repositorio local. Se copian únicamente los tres archivos del modelo:

```
scp -i <llave>.pem modelos/imputer_mice.pkl modelos/baseline.pkl modelos/baseline_meta.json ubuntu@<IP_PUBLICA>:~/ProyectoFinal-FundacionCanguro/modelos/
```

## 7. Levantar los contenedores (terminal de la instancia)

```
cd ~/ProyectoFinal-FundacionCanguro
sudo docker compose -f deploy/docker-compose.prod.yml up -d --build
sudo docker compose -f deploy/docker-compose.prod.yml ps
sudo docker images
```

La primera construcción tarda unos minutos. Para ver los registros se usa sudo docker compose -f deploy/docker-compose.prod.yml logs -f, sin terminales interactivas que bloqueen la sesión.

## 8. Prueba de extremo a extremo

Desde la instancia o desde la máquina local, con la IP pública:

```
bash deploy/smoke_test.sh http://<IP_PUBLICA>:8000
```

El script consulta /health (debe devolver model_ready en true) y envía una predicción con valores sintéticos a /predict. Después, en el navegador:

- http://<IP_PUBLICA>:8000/docs para la documentación de la API.
- http://<IP_PUBLICA>:8501 para el dashboard: completar los tres campos obligatorios y pulsar Validar y generar predicción.

## 9. Evidencia para el informe (ticket 3.5)

Capturas con el usuario de AWS visible en la consola:

- Consola de EC2 con la instancia en Running y las IP privada y pública.
- Terminal SSH con la conexión exitosa.
- Salida de docker images y de docker compose ps, con la IP privada visible en el prompt.
- Navegador con /docs y la IP pública en la barra de direcciones.
- Navegador con el dashboard entregando una predicción a través de la API.
- Salida del script smoke_test.sh.

Junto a cada captura va una explicación breve en español.

## 10. Cierre y control de costos

1. En la instancia, detener los contenedores: sudo docker compose -f deploy/docker-compose.prod.yml down.
2. En la consola de EC2, seleccionar la instancia, Instance state, Terminate.
3. En Canvas, cerrar la sesión del laboratorio y revisar el monto usado.

Una instancia detenida y reiniciada recibe una IP pública nueva, así que las URL del informe solo son válidas durante la sesión en que se tomaron las capturas.

## Problemas conocidos

- La sesión SSH se congela si un contenedor corre en primer plano. Se usa siempre up -d y logs, nunca contenedores interactivos.
- Si la API responde pero model_ready es false, faltan archivos en modelos/ de la instancia o no se copiaron los tres.
- Si scikit-learn del modelo y el de la imagen difieren, aparece InconsistentVersionWarning. El uv.lock fija la versión, así que hay que entrenar con el mismo entorno del repositorio.
- Con t3.micro la API y el dashboard pueden quedarse sin memoria. Se usa t3.small o mayor.
- Las credenciales del laboratorio cambian en cada sesión y no se suben a Git.
