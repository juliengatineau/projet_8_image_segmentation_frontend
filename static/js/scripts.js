function highlightImage(index) {
    // Retirer la classe 'highlight' de toutes les images
    const images = document.querySelectorAll('.image li');
    images.forEach(image => {
        image.classList.remove('highlight');
    });

    // Ajouter la classe 'highlight' à l'image correspondante
    document.getElementById('image-' + index).classList.add('highlight');

    // Retirer la classe 'active' de tous les éléments de la liste
    const items = document.querySelectorAll('.num_id li');
    items.forEach(item => {
        item.classList.remove('active');
    });

    // Ajouter la classe 'active' à l'élément de la liste correspondant
    document.getElementById('num-' + index).classList.add('active');
}


// Activer le premier élément au lancement de la page
document.addEventListener('DOMContentLoaded', () => {
    highlightImage(1);
});



function submitForm(index) {
    const imageElement = document.getElementById('image-' + index).querySelector('img');
    const imageUrl = imageElement.src;

    // Remplir le champ caché avec l'URL de l'image
    document.getElementById('image-url').value = imageUrl;

    // Soumettre le formulaire
    document.getElementById('prediction-form').submit();
}