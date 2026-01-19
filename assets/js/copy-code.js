document.addEventListener('DOMContentLoaded', function() {
  const codeBlocks = document.querySelectorAll('pre.highlight');

  codeBlocks.forEach(function(codeBlock) {
    // Create the copy button
    const copyButton = document.createElement('button');
    copyButton.className = 'copy-code-button';
    copyButton.type = 'button';
    copyButton.innerText = 'Copy';
    copyButton.setAttribute('aria-label', 'Copy code to clipboard');

    // Add click event listener
    copyButton.addEventListener('click', function() {
      const code = codeBlock.querySelector('code').innerText;
      
      navigator.clipboard.writeText(code).then(function() {
        copyButton.innerText = 'Copied!';
        copyButton.classList.add('copied');
        
        setTimeout(function() {
          copyButton.innerText = 'Copy';
          copyButton.classList.remove('copied');
        }, 2000);
      }).catch(function(err) {
        console.error('Failed to copy text: ', err);
        copyButton.innerText = 'Error';
      });
    });

    // Wrap the code block in a container to position the button
    const wrapper = document.createElement('div');
    wrapper.className = 'code-display-wrapper';
    
    // Insert the wrapper before the code block
    codeBlock.parentNode.insertBefore(wrapper, codeBlock);
    
    // Move the code block into the wrapper
    wrapper.appendChild(codeBlock);
    
    // Append the button to the wrapper
    wrapper.appendChild(copyButton);
  });
});
