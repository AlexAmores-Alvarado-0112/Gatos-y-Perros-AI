// ======================================================
// Animal Vision AI
// Funciones del Front-End
// ======================================================


// ================================================
// Vista previa de imagen
// ================================================

function preview(event){

    const archivo = event.target.files[0];

    if(!archivo){
        return;
    }

    // Validar que sea una imagen
    if(!archivo.type.startsWith("image/")){

        alert("Solo se permiten imágenes.");

        event.target.value="";

        return;

    }

    const img=document.getElementById("preview");

    const texto=document.getElementById("upload-text");

    img.src=URL.createObjectURL(archivo);

    img.style.display="block";

    texto.innerHTML="📷 Imagen cargada: <b>"+archivo.name+"</b>";

}



// ================================================
// Animación de barra de confianza
// ================================================

window.addEventListener("load",function(){

    const barra=document.getElementById("barra");

    const valor=document.getElementById("valorConfianza");

    if(!barra || !valor){

        return;

    }

    const porcentaje=parseFloat(valor.value);

    barra.style.width="0%";

    let actual=0;

    const animacion=setInterval(function(){

        if(actual>=porcentaje){

            clearInterval(animacion);

        }else{

            actual++;

            barra.style.width=actual+"%";

        }

    },12);

});




// ================================================
// Drag & Drop
// ================================================

const zona=document.querySelector(".drop");

const input=document.getElementById("image-uploader");

if(zona){

    zona.addEventListener("dragover",function(e){

        e.preventDefault();

        zona.classList.add("drag");

    });

    zona.addEventListener("dragleave",function(){

        zona.classList.remove("drag");

    });

    zona.addEventListener("drop",function(e){

        e.preventDefault();

        zona.classList.remove("drag");

        input.files=e.dataTransfer.files;

        preview({

            target:input

        });

    });

}




// ================================================
// Loader al enviar imagen
// ================================================

const formulario=document.querySelector("form");

if(formulario){

    formulario.addEventListener("submit",function(){

        const boton=document.querySelector("button");

        boton.disabled=true;

        boton.innerHTML="⏳ Analizando imagen...";

    });

}




// ================================================
// Cambiar color de la barra según confianza
// ================================================

window.addEventListener("load",function(){

    const barra=document.getElementById("barra");

    const valor=document.getElementById("valorConfianza");

    if(!barra || !valor){

        return;

    }

    const confianza=parseFloat(valor.value);

    if(confianza>=90){

        barra.style.background="#27ae60";

    }

    else if(confianza>=70){

        barra.style.background="#f39c12";

    }

    else{

        barra.style.background="#e74c3c";

    }

});
