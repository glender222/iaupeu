# Despliegue automático de la demo

Los cambios en `unidad3/web/` se publican al hacer push a `main`. GitHub Actions ejecuta `fraude-ci.yml` para comprobar la sintaxis, ambos modelos, el tratamiento de importes negativos y el CSV. La VM consulta cada dos minutos el último commit de `main` y lo instala solamente cuando la ejecución de CI de ese mismo commit ha terminado correctamente. Un PR también ejecuta las pruebas, pero no despliega hasta llegar a `main`.

La demo se publica en la ruta `/fraude/` del túnel existente. La actualización cambia un enlace a la versión estática, sin reiniciar LinkenJobs ni el túnel. La dirección puede cambiar si el Quick Tunnel se reinicia. La VM debe estar encendida y tener acceso a GitHub; si hay fallos de red, límite de API o CI no aprobado, conserva la última versión y vuelve a intentarlo.

Para revisar el proceso en la VM: `systemctl status iaupeu-fraude.timer` y `journalctl -u iaupeu-fraude-deploy.service -n 50 --no-pager`. La versión activa se registra en `/home/azureuser/iaupeu-fraude/version-activa`. El servicio de actualización usa `actualizar.sh` instalado en la VM; modificar el archivo del repositorio no actualiza automáticamente el propio servicio de despliegue.
